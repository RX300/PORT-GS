"""Native gsplat camera rendering and geometry-derived shadow hints."""

import torch
from torch.nn import functional as F
from gsplat import rasterization
from gsplat.utils import normalized_quat_to_rotmat


def light_view(light, center):
    forward = F.normalize(center - light, dim=0)
    # Select the coordinate axis least parallel to the viewing direction.
    up = torch.eye(3, device=light.device)[forward.abs().argmin()]
    right = F.normalize(torch.linalg.cross(forward, up), dim=0)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack((right, down, forward))
    matrix = torch.eye(4, device=light.device)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ light
    return matrix


def _deep_map(inputs, view, camera, light, radius, focal, depth_near, depth_far, query=None):
    resolution, bins = 256, 64
    # The adaptive sampling grid is fixed for this derivative evaluation.
    depth_near, depth_far = depth_near.detach(), depth_far.detach()
    k = torch.eye(3, device=light.device)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = k[1, 2] = resolution * 0.5

    depth_width = depth_far - depth_near
    bin_position = (camera[:, 2] - depth_near) / depth_width * bins - 0.5
    bin_position = bin_position.clamp(0, bins - 1)
    lower = bin_position.floor().long()
    upper = (lower + 1).clamp(max=bins - 1)
    upper_weight = bin_position - lower
    bin_colors = torch.zeros((len(camera), bins), device=camera.device, dtype=camera.dtype)
    bin_colors.scatter_add_(1, lower[:, None], (1 - upper_weight)[:, None])
    bin_colors.scatter_add_(1, upper[:, None], upper_weight[:, None])
    accumulated, _, _ = rasterization(
        **inputs,
        colors=bin_colors,
        viewmats=view[None],
        Ks=k[None],
        width=resolution,
        height=resolution,
        backgrounds=torch.zeros((1, bins), device=camera.device),
        packed=False,
    )
    prefix = accumulated[0].cumsum(-1).clamp(0, 1)
    volume = prefix.permute(2, 0, 1)[None, None]
    if query is not None:
        camera = camera[query]
        inputs = {name: value[query] for name, value in inputs.items()}

    bin_width = depth_width / bins
    # Self-occlusion exclusion is a numerical offset, not an appearance parameter.
    with torch.no_grad():
        rotation = normalized_quat_to_rotmat(F.normalize(inputs["quats"], dim=-1))
        ray_origin = inputs["means"] - light
        ray_length = ray_origin.norm(dim=-1)
        ray = ray_origin / ray_length[:, None]
        local_ray = (rotation.transpose(1, 2) @ ray[..., None]).squeeze(-1)
        sigma = torch.rsqrt((local_ray.square() / inputs["scales"].square()).sum(-1))
        sigma_z = sigma * camera[:, 2] / ray_length
        bias = (2 * sigma_z).clamp_min(1.5 * bin_width) + radius * 0.002
    query_depth = camera[:, 2] - bias
    xy = camera[:, :2] / camera[:, 2:] * focal + resolution * 0.5
    grid_xy = xy / resolution * 2 - 1
    # With align_corners=False, this maps the near/far endpoints to the
    # outer faces and each bin center to its exact voxel center.
    below_near = query_depth <= depth_near
    grid_depth = query_depth.clamp(depth_near, depth_far)
    grid_z = 2 * (grid_depth - depth_near) / depth_width - 1
    grid = torch.stack((grid_xy[:, 0], grid_xy[:, 1], grid_z), dim=-1)
    grid = grid.reshape(1, 1, 1, len(camera), 3)
    queried = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0, 0, 0]
    queried = queried.masked_fill(below_near, 0.0)
    return (1 - queried).clamp(0, 1)


def _deep_cube(inputs, light, radius):
    """Six point-light faces; every Gaussian receives a visibility query.

    All Gaussians are rasterized into every requested face, including splats
    overlapping its boundary. One-pixel face padding avoids lookup seams.
    This remains a finite-bin Gaussian shadow approximation.
    """
    delta = inputs["means"] - light
    axis = delta.abs().argmax(-1)
    sign = delta.gather(1, axis[:, None]).squeeze(1) < 0
    face = axis + sign.long() * 3
    directions = torch.cat((torch.eye(3, device=light.device), -torch.eye(3, device=light.device)))
    support = inputs["scales"].max(-1).values[:, None] * 3
    visibility = torch.empty(len(delta), device=light.device, dtype=delta.dtype)
    for index, direction in enumerate(directions):
        query = face == index
        if not query.any():
            continue
        view = light_view(light, light + direction)
        camera = inputs["means"] @ view[:3, :3].T + view[:3, 3]
        front = camera[:, 2] > 0
        near = (camera[:, 2:] - support)[front].min().clamp_min(0)
        far = (camera[:, 2:] + support)[front].max()
        visibility[query] = _deep_map(inputs, view, camera, light, radius, 127.0, near, far, query)
    return visibility


def visibility_hint(gaussians, light, resolution=128, mode="depth"):
    """One perspective map fitted to the entire object, for exterior point lights.

    Deep shadows use six padded cube faces when a fitted single view cannot
    contain every Gaussian support. The external-view path stays unchanged.
    ``depth`` uses a single-depth map; ``deep`` uses a 64-bin shadow volume.
    Gradients reach occluder geometry and opacity, and receiver coordinates.
    Adaptive sampling bounds and self-exclusion offsets stay fixed in backward.
    """
    if mode not in {"depth", "deep"}:
        raise ValueError(f"unknown shadow mode: {mode}")
    inputs = gaussians.raster_inputs()
    view = light_view(light, gaussians.center)
    camera = inputs["means"] @ view[:3, :3].T + view[:3, 3]
    support = inputs["scales"].max(-1).values[:, None] * 3

    external = (camera[:, 2:] - support).min() > 0
    if mode == "deep":
        if not external:
            return _deep_cube(inputs, light, gaussians.radius)
        # gsplat has no K derivatives: map and query share a fixed focal length.
        extent = ((camera[:, :2].abs() + support) / (camera[:, 2:] - support)).max().detach() * 1.05
        return _deep_map(
            inputs,
            view,
            camera,
            light,
            gaussians.radius,
            128.0 / extent,
            (camera[:, 2:] - support).min(),
            (camera[:, 2:] + support).max(),
        )
    assert external, "Exterior-light shadow map contract violated"

    extent = ((camera[:, :2].abs() + support) / (camera[:, 2:] - support)).max().detach() * 1.05
    focal = resolution * 0.5 / extent
    k = torch.eye(3, device=light.device)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = k[1, 2] = resolution * 0.5
    depth, alpha, _ = rasterization(
        **inputs,
        colors=torch.ones_like(inputs["means"][:, :1]),
        viewmats=view[None],
        Ks=k[None],
        width=resolution,
        height=resolution,
        render_mode="ED",
        packed=False,
    )
    xy = camera[:, :2] / camera[:, 2:] * focal + resolution * 0.5
    # gsplat rasterizes pixel centers at (j+.5, i+.5).
    grid = (xy / resolution * 2 - 1)[None, None]
    sampled = F.grid_sample(depth.permute(0, 3, 1, 2), grid, align_corners=False)[0, 0, 0]
    coverage = F.grid_sample(alpha.permute(0, 3, 1, 2), grid, align_corners=False)[0, 0, 0]
    bias = (inputs["scales"].max(-1).values * 2 + gaussians.radius * 0.01).detach()
    visible = torch.sigmoid((sampled + bias - camera[:, 2]) / (gaussians.radius * 0.015))
    return (1 - coverage) + coverage * visible


def render(
    gaussians,
    transport,
    sample,
    background=1.0,
    shadow=False,
    port_active=True,
    shadow_mode="depth",
    absgrad=False,
):
    h, w = sample["image"].shape[:2]
    means = gaussians.params["means"]
    visibility = (
        visibility_hint(gaussians, sample["light_pos"], mode=shadow_mode)
        if shadow
        else torch.ones(len(means), device=means.device)
    )
    attributes = torch.cat(
        (gaussians.params["base"], gaussians.params["features"], visibility[:, None]), dim=-1
    )
    rendered, alpha, info = rasterization(
        **gaussians.raster_inputs(),
        colors=attributes,
        viewmats=sample["viewmat"][None],
        Ks=sample["K"][None],
        width=w,
        height=h,
        backgrounds=attributes.new_zeros((1, attributes.shape[-1])),
        render_mode="RGB+ED",
        # One attribute/depth pass keeps gsplat's absgrad statistic complete.
        channel_chunk=attributes.shape[-1] + 1,
        packed=False,
        absgrad=absgrad,
    )
    alpha = alpha[0]
    covered = alpha[..., 0] > 0
    # RGB+ED normalizes only the final camera-Z channel; attributes retain alpha weights.
    pixel_attributes = rendered[0, ..., :-1][covered] / alpha[covered]
    depth = rendered[0, ..., -1][covered][:, None]
    y, x = torch.meshgrid(
        torch.arange(h, device=means.device, dtype=means.dtype) + 0.5,
        torch.arange(w, device=means.device, dtype=means.dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    camera_points = (pixels[covered] @ torch.linalg.inv(sample["K"]).T) * depth
    viewmat = sample["viewmat"]
    world_points = (camera_points - viewmat[:3, 3]) @ viewmat[:3, :3]
    receivers = {
        "means": world_points,
        "base": pixel_attributes[:, :3],
        "features": pixel_attributes[:, 3:-1],
        "visibility": pixel_attributes[:, -1],
    }
    foreground = means.new_zeros((h, w, 3))
    foreground[covered] = transport(
        gaussians,
        receivers,
        sample["c2w"][:3, 3],
        sample["light_pos"],
        sample["light_intensity"],
        visibility,
        port_active,
    )
    return foreground * alpha + background * (1 - alpha), alpha, info

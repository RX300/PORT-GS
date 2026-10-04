"""Native gsplat camera rendering and geometry-derived shadow hints."""

import torch
from torch.nn import functional as F
from gsplat import rasterization, rasterization_2dgs
from gsplat.utils import normalized_quat_to_rotmat
from gsplat.cuda._wrapper import rasterize_to_indices_in_range_2dgs


_SURFACE_RAY_BATCH_SIZE = 4096


def intersection_depth(info):
    """Expected ray/surfel depth with gsplat's existing visibility and 2D filter.

    This differentiable sparse reference keeps the native center-depth ordering.
    The screen-space low-pass branch uses center Z, as its support is not on the
    surfel plane. Native distortion remains a center-depth proxy.
    """
    w, h = info['width'], info['height']
    means2d, transforms = info['means2d'], info['ray_transforms']
    with torch.no_grad():
        gid, pid, iid = rasterize_to_indices_in_range_2dgs(
            0, len(info['flatten_ids'])+1, means2d.new_ones((1, h, w)),
            means2d, transforms, info['opacities'], w, h, info['tile_size'],
            info['isect_offsets'], info['flatten_ids'])
        # The stable order retains native front-to-back order inside every ray.
        order = pid.argsort(stable=True)
        gid, pid, iid = gid[order], pid[order], iid[order]
        starts = torch.nonzero(torch.cat((pid.new_ones(1, dtype=torch.bool),
                                         pid[1:] != pid[:-1]))).flatten()
        lengths = torch.diff(torch.cat((starts, starts.new_tensor([len(pid)]))))
        groups = torch.repeat_interleave(torch.arange(len(starts), device=pid.device),
                                        lengths)
    if not len(pid):
        return transforms.sum()*0 + means2d.new_zeros((1, h, w, 1))
    px, py = pid % w + .5, pid // w + .5
    matrix = transforms[iid, gid]
    cross = torch.linalg.cross(px[:, None]*matrix[:, 2]-matrix[:, 0],
                               py[:, None]*matrix[:, 2]-matrix[:, 1])
    uv = cross[:, :2] / cross[:, 2:]
    rho3 = uv.square().sum(-1)
    rho2 = 2*(means2d[iid, gid]-torch.stack((px, py), -1)).square().sum(-1)
    a = (info['opacities'][iid, gid]*torch.exp(-.5*torch.minimum(rho3, rho2))).clamp(max=.999)
    # Float64 prevents loss of per-ray precision in a global segmented scan.
    log_t = torch.log1p(-a.double())
    prefix = log_t.cumsum(0)-log_t
    weights = a*torch.exp(prefix-prefix[starts[groups]]).to(a.dtype)
    z = (uv*matrix[:, 2, :2]).sum(-1)+matrix[:, 2, 2]
    z = torch.where(rho3 <= rho2, z, info['depths'][iid, gid])
    # Segmented reduction has a fixed summation order; atomic index_add does not.
    sums = torch.segment_reduce(weights[:, None]*torch.stack((z, torch.ones_like(z)), -1),
                                reduce='sum', lengths=lengths)
    depth = sums[:, :1]/sums[:, 1:].clamp_min(1e-10)
    return means2d.new_zeros((h*w, 1)).index_copy(0, pid[starts], depth).reshape(1, h, w, 1)


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


def _rasterize(geometry, surface_depth="center", **kwargs):
    if geometry == "3dgs":
        return rasterization(**kwargs)
    kwargs.pop("channel_chunk", None)
    mode = kwargs.get("render_mode", "RGB")
    channels = kwargs["colors"].shape[-1]
    with_depth = mode in ("RGB+D", "RGB+ED")
    total = channels + int(with_depth) if mode.startswith("RGB") else 1
    # gsplat 1.5.3 pads unsupported 2DGS channel counts with torch.empty.
    # NaN padding poisons geometry gradients even though those channels are discarded.
    padding = 0 if total in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512) else (1 << (total-1).bit_length()) - total
    if padding:
        kwargs["colors"] = F.pad(kwargs["colors"], (0, padding))
        if kwargs.get("backgrounds") is not None:
            kwargs["backgrounds"] = F.pad(kwargs["backgrounds"], (0, padding))
    # gsplat 1.5.3 requires an explicit camera axis for 2DGS feature colors.
    kwargs["colors"] = kwargs["colors"][None]
    surface = mode == "RGB+ED"
    colors, alpha, normals, _, distortion, _, info = rasterization_2dgs(
        **kwargs, distloss=surface, depth_mode="expected")
    if padding:
        colors = (torch.cat((colors[..., :channels], colors[..., -1:]), dim=-1)
                  if with_depth else colors[..., :channels])
    if surface:
        if surface_depth == 'intersection':
            colors = torch.cat((colors[..., :-1], intersection_depth(info)), dim=-1)
        info["surface_normals"] = normals[0]  # world coordinates, alpha weighted
        info["surface_depth"] = colors[0, ..., -1:]
        info["surface_distortion"] = distortion[0]
    return colors, alpha, info


def _deep_map(inputs, view, camera, light, radius, focal, depth_near, depth_far, query=None, geometry="3dgs"):
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
    accumulated, _, _ = _rasterize(
        geometry,
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
        if geometry == "2dgs":
            # Disks have no thickness. Exclude the two-bin center-depth deposit.
            # This is a finite-bin shadow approximation, not ray/surfel tracing.
            bias = 2 * bin_width + radius * 0.002
        else:
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


def _deep_cube(inputs, light, radius, geometry="3dgs"):
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
    support = inputs["scales"][:, :2 if geometry == "2dgs" else 3].max(-1).values[:, None] * 3
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
        visibility[query] = _deep_map(inputs, view, camera, light, radius, 127.0, near, far, query, geometry)
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
    geometry = gaussians.geometry
    view = light_view(light, gaussians.center)
    camera = inputs["means"] @ view[:3, :3].T + view[:3, 3]
    support = inputs["scales"][:, :2 if geometry == "2dgs" else 3].max(-1).values[:, None] * 3

    external = (camera[:, 2:] - support).min() > 0
    if mode == "deep":
        if not external:
            return _deep_cube(inputs, light, gaussians.radius, geometry)
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
            geometry=geometry,
        )
    if not external:
        raise RuntimeError("Depth shadow maps require an exterior light; use --shadow-mode deep")

    extent = ((camera[:, :2].abs() + support) / (camera[:, 2:] - support)).max().detach() * 1.05
    focal = resolution * 0.5 / extent
    k = torch.eye(3, device=light.device)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = k[1, 2] = resolution * 0.5
    depth, alpha, _ = _rasterize(
        geometry,
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
    bias = (inputs["scales"][:, :2 if geometry == "2dgs" else 3].max(-1).values * 2 + gaussians.radius * 0.01).detach()
    visible = torch.sigmoid((sampled + bias - camera[:, 2]) / (gaussians.radius * 0.015))
    return (1 - coverage) + coverage * visible


def apply_radiance_residual(foreground, alpha, receivers, covered, transport, sample,
                            radiance_residual, residual_indices=None):
    """Correct unique covered pixels; repeated loss indices keep their weights."""
    foreground, alpha = foreground.detach().clone(), alpha.detach()
    covered_ids = covered.reshape(-1).nonzero().flatten()
    if residual_indices is None:
        query_rows = torch.arange(len(covered_ids), device=covered.device)
        requested_pixels = covered.numel()
        requested_covered = len(covered_ids)
    else:
        selected = residual_indices.unique()
        selected = selected[covered.reshape(-1)[selected]]
        query_rows = torch.searchsorted(covered_ids, selected)
        requested_pixels = residual_indices.numel()
        requested_covered = covered.reshape(-1)[residual_indices].sum()
    frozen_receivers = {key: value.detach() for key,value in receivers.items()}
    normals = (transport.material_normal(frozen_receivers) if radiance_residual.normal_source == 'material'
               else frozen_receivers['normals'])
    flat = foreground.reshape(-1, 3)
    clamped = foreground.new_zeros(())
    delta_abs = foreground.new_zeros(())
    base_abs = foreground.new_zeros(())
    interaction_stats = ({f'interaction_{name}_{moment}_mean': foreground.new_zeros(())
                          for name in ('grid', 'spatial', 'angular', 'output')
                          for moment in ('abs', 'square')}
                         if radiance_residual.interaction != 'none' else {})
    if radiance_residual.angular_bank != 'none':
        interaction_stats.update({f'angular_kernel_{index}_{moment}': foreground.new_zeros(())
                                  for index in range(8)
                                  for moment in ('mean', 'square_mean', 'zero_fraction')})
        interaction_stats.update({name: foreground.new_zeros(()) for name in (
            'center_rotation_deg_mean', 'center_rotation_deg_square_mean', 'center_raw_norm_min',
            'center_raw_norm_below_01_fraction', 'center_offset_square_mean',
            'angular_output_square_mean')})
    if len(query_rows):
        for chunk_index, rows in enumerate(query_rows.split(4096)):
            pixels = covered_ids[rows]
            base = flat[pixels].detach()
            delta = radiance_residual(frozen_receivers['means'][rows], normals[rows].detach(),
                                      sample['c2w'][:3, 3].detach(), sample['light_pos'].detach(),
                                      sample['light_intensity'].detach(), transport.light_scale.detach(),
                                      return_stats=bool(interaction_stats))
            if interaction_stats:
                delta, chunk_stats = delta
                for key, value in chunk_stats.items():
                    if key == 'center_raw_norm_min':
                        interaction_stats[key] = (value if chunk_index == 0
                                                  else torch.minimum(interaction_stats[key], value))
                    else:
                        interaction_stats[key] += value*len(rows)
            combined = base+delta
            flat[pixels] = combined.clamp_min(0)
            clamped += (combined.detach() < 0).sum()
            delta_abs += delta.detach().abs().sum()
            base_abs += base.abs().sum()
        channels = 3*len(query_rows)
        clamped, delta_abs, base_abs = clamped/channels, delta_abs/channels, base_abs/channels
        interaction_stats = {key: value if key == 'center_raw_norm_min' else value/len(query_rows)
                             for key,value in interaction_stats.items()}
    else:
        # No sampled surface means zero correction and a valid zero gradient.
        foreground = foreground+radiance_residual.network[-1].weight.sum()*0
    stats = {'queried_pixels':len(query_rows), 'covered_pixels':len(covered_ids),
             'requested_pixels':requested_pixels, 'requested_covered_pixels':requested_covered,
             'clamp_fraction':clamped, 'delta_abs_mean':delta_abs, 'base_abs_mean':base_abs}
    stats.update(interaction_stats)
    return foreground, alpha, stats


def covariance_normals(gaussians, eye):
    """Face-forward shortest covariance-axis proxy for a 3D Gaussian material.

    This is not the continuous-depth normal of the GGGS author renderer.
    Axis choice and facing sign are discrete; rotations retain gradients.
    """
    p=gaussians.params
    rotation=normalized_quat_to_rotmat(F.normalize(p['quats'],dim=-1))
    axis=p['scales'].argmin(-1)
    normal=rotation.gather(2,axis[:,None,None].expand(-1,3,1)).squeeze(-1)
    return normal*torch.where(((eye-p['means'])*normal).sum(-1,keepdim=True)>=0,1.,-1.)


def render(
    gaussians,
    transport,
    sample,
    background=1.0,
    shadow=False,
    port_active=True,
    shadow_mode="depth",
    absgrad=False,
    geometry_only=False,
    surface_field=None,
    normal_field=None,
    radiance_residual=None,
    residual_indices=None,
    appearance_weight=None,
):
    """Render a full HWC image.

    ``appearance_weight`` (HW1, training only) scales the gradient that reaches the
    foreground radiance per pixel; the rest of the pixel's gradient reaches coverage only.
    """
    h, w = sample["image"].shape[:2]
    means = gaussians.params["means"]
    # Light-space methods shade visibility per receiver from their own light pass, unless
    # they explicitly request the per-Gaussian deep shadow (an ablation of light_atlas).
    light_space = getattr(transport, "light_space", False)
    gaussian_shadow = not light_space or getattr(transport, "per_gaussian_visibility", False)
    visibility = (
        visibility_hint(gaussians, sample["light_pos"], mode=shadow_mode)
        if shadow and not geometry_only and gaussian_shadow
        else torch.ones(len(means), device=means.device)
    )
    if geometry_only:
        # Degree-zero RGB splatting, independent of lights and all transport networks.
        attributes = F.softplus(gaussians.params["base"])
    else:
        attributes = torch.cat(
            (gaussians.params["base"], gaussians.params["features"], visibility[:, None]), dim=-1
        )
    normal3d = (not geometry_only and gaussians.geometry == '3dgs'
                and getattr(transport,'requires_normals',False))
    if normal3d:
        attributes=torch.cat((attributes,covariance_normals(gaussians,sample['c2w'][:3,3])),dim=-1)
    rendered, alpha, info = _rasterize(
        gaussians.geometry,
        surface_depth=gaussians.surface_depth,
        **gaussians.raster_inputs(),
        colors=attributes,
        viewmats=sample["viewmat"][None],
        Ks=sample["K"][None],
        width=w,
        height=h,
        backgrounds=attributes.new_zeros((1, attributes.shape[-1])),
        render_mode="RGB+ED",
        # Native absgrad measures the CUDA attribute path. Sparse intersection
        # depth has geometry gradients, but adds no per-pixel absgrad statistic.
        channel_chunk=attributes.shape[-1] + 1,
        packed=False,
        absgrad=absgrad,
    )
    alpha = alpha[0]
    if geometry_only:
        return rendered[0, ..., :3] + background * (1 - alpha), alpha, info
    covered = alpha[..., 0] > 0
    # RGB+ED normalizes only the final camera-Z channel; attributes retain alpha weights.
    pixel_attributes = rendered[0, ..., :-1][covered] / alpha[covered]
    if normal3d:
        pixel_normals=F.normalize(pixel_attributes[:,-3:],dim=-1)
        pixel_attributes=pixel_attributes[:,:-3]
        info['surface_normals']=rendered[0,...,-4:-1]
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
    if gaussians.geometry == "2dgs":
        receivers["normals"] = F.normalize(info["surface_normals"][covered], dim=-1)
    elif normal3d:
        receivers['normals']=pixel_normals
    if surface_field is not None:
        receivers["normals"] = surface_field.shading_normals(world_points, sample["c2w"][:3, 3])
    if normal_field is not None:
        receivers['normal_residual'] = normal_field(world_points)
        info['normal_residual'] = receivers['normal_residual']
    foreground = means.new_zeros((h, w, 3))
    foreground[covered] = transport(
        gaussians,
        receivers,
        sample["c2w"][:3, 3],
        sample["light_pos"],
        sample["light_intensity"],
        visibility,
        port_active,
        **({"shadow": shadow} if light_space else {}),
    )
    if radiance_residual is not None:
        foreground, alpha, info['residual_stats'] = apply_radiance_residual(
            foreground, alpha, receivers, covered, transport, sample, radiance_residual, residual_indices)
    if appearance_weight is not None:
        foreground = foreground * appearance_weight + foreground.detach() * (1 - appearance_weight)
    return foreground * alpha + background * (1 - alpha), alpha, info

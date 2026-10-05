"""Fit-frame camera and light corrections, and the scene-wide light scale, during training.

Held-out frames never use these corrections. In frozen stages (radiance
residual, SDF-volume-only) saved camera corrections are applied but have no
optimizer, so no learning rate or step is needed.
"""

import torch

from training.schedule import decayed_rate


class CameraFit:
    def __init__(self, args, config, saved, samples, fit_indices, gaussians, trainable):
        from cameras import build_camera_offsets

        restored = args.init_checkpoint and saved["camera_offsets"] is not None
        if restored:
            saved_mode = saved["config"].get("camera_mode", "anchor")
            if (args.sdf_volume_only or args.radiance_residual) and saved_mode != args.camera_mode:
                args.camera_mode = config["camera_mode"] = saved_mode
            if saved_mode != args.camera_mode:
                raise ValueError("Resuming saved camera corrections requires their --camera-mode")
        self.offsets = build_camera_offsets(args.camera_mode, len(fit_indices), gaussians.radius).to(gaussians.center.device)
        if restored:
            assert saved["fit_indices"] == fit_indices
            self.offsets.load_state_dict(saved["camera_offsets"])
        self.optimizer = self.offsets.optimizer(args.camera_lr) if trainable else None
        if not trainable:
            self.offsets.requires_grad_(False)
        self.indices = {index: local for local, index in enumerate(fit_indices)}
        self.lr, self.lr_final = args.camera_lr, args.camera_lr_final
        self.start = args.camera_start
        self.steps = args.steps if args.lr_decay_steps is None else args.lr_decay_steps
        # LiSA's light-normalized geometry stage still needs the real scenes'
        # train-camera calibration at its configured start iteration.
        self.geometry_stage = config.get('representation') == 'light_atlas'
        self.gauge = trainable and args.camera_gauge == 'translation'
        if self.gauge:
            self.offsets.set_translation_gauge(torch.stack([samples[i]["viewmat"] for i in fit_indices]),
                                               torch.stack([samples[i]["K"] for i in fit_indices]),
                                               gaussians.center)

    @property
    def trainable(self):
        return self.optimizer is not None

    def active(self, step, geometry_only):
        return (not geometry_only or self.geometry_stage) and step >= self.start

    def begin(self, step, sample, sample_index):
        """Reset gradients, schedule the rate, and return the corrected fit sample."""
        if self.trainable:
            self.optimizer.zero_grad(set_to_none=True)
            if self.lr_final is not None:
                self.optimizer.param_groups[0]["lr"] = decayed_rate(
                    self.lr, self.lr_final, min(step, self.steps), self.start, self.steps)
        return self.offsets.correct(sample, self.indices[sample_index])

    def regularization(self, sample_index):
        return self.offsets.regularization(self.indices[sample_index])

    def step(self, gaussians):
        """Step the corrections; return the compensating scene shift when the gauge is fixed."""
        self.optimizer.step()
        if not self.gauge:
            return None
        shift = self.offsets.remove_translation_gauge()
        with torch.no_grad():
            gaussians.params["means"] += shift
        return shift

    def log_fields(self):
        lr = self.optimizer.param_groups[0]['lr'] if self.trainable else self.lr
        return {'camera_rotation_rms': self.offsets.rms()[0], 'camera_lr': lr}


class LightFit:
    def __init__(self, args, saved, fit_indices, gaussians):
        from cameras import TrainLightOffsets

        self.offsets = TrainLightOffsets(len(fit_indices), gaussians.radius).to(gaussians.center.device)
        if args.init_checkpoint and saved.get("light_offsets") is not None:
            assert saved["fit_indices"] == fit_indices
            self.offsets.load_state_dict(saved["light_offsets"])
        self.optimizer = self.offsets.optimizer(args.light_lr)
        self.indices = {index: local for local, index in enumerate(fit_indices)}
        self.lr, self.lr_final = args.light_lr, args.light_lr_final
        self.start, self.steps = args.light_start, args.steps

    def active(self, step, geometry_only):
        return not geometry_only and step >= self.start

    def begin(self, step, sample, sample_index):
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.param_groups[0]["lr"] = decayed_rate(self.lr, self.lr_final, step, self.start, self.steps)
        return self.offsets.correct(sample, self.indices[sample_index])

    def regularization(self, sample_index):
        return self.offsets.regularization(self.indices[sample_index])

    def step(self):
        self.optimizer.step()

    def log_fields(self):
        return {'light_offset_rms': self.offsets.rms(), 'light_lr': self.optimizer.param_groups[0]['lr']}


class LightScaleFit:
    """One positive scene-wide irradiance normalization.

    The checkpoint/render contract keeps the fitted scale in the transport's
    existing buffer; only training needs the log parameter.
    """

    def __init__(self, transport):
        self.initial = transport.light_scale.detach().clone()
        self.log_scale = torch.nn.Parameter(self.initial.log())
        self.optimizer = torch.optim.Adam([self.log_scale], lr=.001)

    def begin(self, transport):
        self.optimizer.zero_grad(set_to_none=True)
        transport.light_scale = self.log_scale.exp()

    def step(self, transport):
        self.optimizer.step()
        transport.light_scale = self.log_scale.detach().exp()

    def log_fields(self, transport):
        return {'light_scale': float(transport.light_scale),
                'relative_light_gain': float(self.initial / transport.light_scale)}

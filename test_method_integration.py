"""Real Cat CLI training, checkpoint reload and evaluation for every method.

Run explicitly on one idle GPU; all evidence is written beneath --output.
"""
import argparse
import json
import math
import random
import subprocess
import sys
import tarfile
from pathlib import Path

import torch

from evaluate import load_model, load_surface_field, load_normal_field, load_sdf_volume, render_observation, to_device, target_image
from data import SceneDataset, _frame_path
from methods import METHODS, resolve_config
from gaussians import Gaussians
from renderer import visibility_hint, _rasterize, rasterization_2dgs
from surface import SurfacePriors, surface_losses
from refinement import Refinement
from training.schedule import set_training_stage


def check_radiance_residual_checkpoint(checkpoint, output, normal_source='material', residual_frames=None,
                                      interaction='none', angular_bank='none', paired_context=False, pair_weight=0.):
    """Actual fixed-GS receiver path, then native-resolution CLI fit/resume."""
    import gc
    from cameras import TrainCameraOffsets
    from evaluate import load_radiance_residual, observation_image
    from make_validation_manifest import train_argv
    from materials.radiance_residual import RadianceResidual
    from renderer import render
    torch.set_num_threads(8)
    torch.manual_seed(0)
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).parent
    with tarfile.open(output/'source.tar', 'w') as archive:
        for name in ('test_method_integration.py', 'train.py', 'evaluate.py', 'renderer.py',
                     'materials/radiance_residual.py', 'materials/spatial_detail.py',
                     'make_validation_manifest.py', 'residual_sampling.py'):
            archive.add(project/name, arcname=name)
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = resolve_config(source['config'])
    assert cfg['representation'] == 'neural_material'
    assert not cfg.get('radiance_residual', False)
    report = {'source':str(checkpoint), 'normal_source':normal_source, 'interaction':interaction, 'angular_bank':angular_bank,
              'paired_context':paired_context, 'pair_weight':pair_weight,
              'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__,
              'source_fit_frames':len(source['fit_indices']), 'render_resolution':128,
              'training_resolution':512, 'residual_rays':2048,
              'peak_fraction':.25, 'context_fraction':.25, 'seed':0,
              'residual_frames':residual_frames,
              'scope':'integration contracts only; not a relighting quality experiment'}
    g, model, saved = load_model(checkpoint)
    sample = to_device(SceneDataset(cfg['scene'], 'train', 128,
        unit_light_intensity=cfg['unit_light_intensity'])[source['fit_indices'][1]], 'cuda')
    offsets = None
    if source['camera_offsets'] is not None:
        offsets = TrainCameraOffsets(len(source['fit_indices']), g.radius).cuda()
        offsets.load_state_dict(saved['camera_offsets'])
        sample = offsets.correct(sample, 1)
    shadow = source['step'] >= cfg['shadow_start']
    port_active = source['step'] >= cfg['port_start']
    zero_checks = {}
    for mode in (('geometry', 'material') if angular_bank == 'none' else ('geometry',)):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            head = RadianceResidual(g.center, g.radius, normal_source=mode, interaction=interaction, angular_bank=angular_bank)
        with torch.no_grad():
            original, original_alpha, _ = render(g, model, sample, cfg['background'], shadow, port_active,
                                               shadow_mode=cfg['shadow_mode'])
            corrected, alpha, _ = render(g, model, sample, cfg['background'], shadow, port_active,
                shadow_mode=cfg['shadow_mode'], radiance_residual=head)
            observed = observation_image(original, cfg['display_gamma'],
                alpha=None if sample['is_hdr'] else original_alpha, background=cfg['background'])
            predicted, observed_alpha, _ = render_observation(g, model, sample, cfg['background'],
                shadow, port_active, cfg['display_gamma'], cfg['shadow_mode'], radiance_residual=head)
        torch.testing.assert_close(corrected, original, rtol=0, atol=0)
        torch.testing.assert_close(alpha, original_alpha, rtol=0, atol=0)
        torch.testing.assert_close(observed_alpha, original_alpha, rtol=0, atol=0)
        torch.testing.assert_close(predicted, observed, rtol=0, atol=0)
        zero_checks[mode] = {'linear':True, 'observation':True, 'alpha':True}
        with torch.no_grad():
            head.network[-1].bias.copy_(head.network[-1].bias.new_tensor([.02, -.02, .01]))
            nonzero, _, _ = render(g, model, sample, cfg['background'], shadow, port_active,
                shadow_mode=cfg['shadow_mode'], radiance_residual=head)
            doubled, _, _ = render(g, model, dict(sample, light_intensity=sample['light_intensity']*2),
                cfg['background'], shadow, port_active, shadow_mode=cfg['shadow_mode'], radiance_residual=head)
            dark, dark_alpha, _ = render(g, model, dict(sample, light_intensity=sample['light_intensity']*0),
                cfg['background'], shadow, port_active, shadow_mode=cfg['shadow_mode'], radiance_residual=head)
        background = cfg['background']*(1-original_alpha)
        torch.testing.assert_close(doubled-background, 2*(nonzero-background), rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(dark, (cfg['background']*(1-dark_alpha)).expand_as(dark), rtol=0, atol=0)
        assert (nonzero[..., 0]-original[..., 0]).max()>0
        assert (nonzero[..., 1]-original[..., 1]).min()<0
        zero_checks[mode].update(signed_correction=True, linear_light_homogeneity=True, zero_intensity=True)
        # Force every parent parameter trainable to test detach boundaries, not
        # merely a training loop that happens to disable parent gradients.
        for parameter in list(g.parameters())+list(model.parameters()):
            parameter.requires_grad_(True)
        g.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        ids = (original_alpha[...,0]>.9).reshape(-1).nonzero().flatten()[:256]
        assert len(ids)>0
        image, _, info = render_observation(g, model, sample, cfg['background'], shadow, port_active,
            cfg['display_gamma'], cfg['shadow_mode'], radiance_residual=head, residual_indices=ids)
        image.reshape(-1, 3)[ids].mean().backward()
        assert all(parameter.grad is None for parameter in g.parameters())
        assert all(parameter.grad is None for parameter in model.parameters())
        assert offsets is None or offsets.raw.grad is None
        gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients)>0
        zero_checks[mode]['parent_and_camera_gradients_isolated'] = True
        del head, original, original_alpha, corrected, alpha, observed, predicted, observed_alpha
        del nonzero, doubled, dark, dark_alpha, background, ids, image, info, gradients
    report['render_contracts'] = zero_checks
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: real residual zero rendering, signed correction, light homogeneity and gradient isolation', flush=True)
    keys = set(METHODS['neural_material'].cli_fields) | {
        'scene', 'representation', 'background', 'display_gamma', 'shadow_mode', 'max_points', 'fit_all'}
    options = {key.replace('_', '-'):cfg[key] for key in keys}
    options.update({'steps':3, 'lr-decay-steps':3, 'resolution':512, 'seed':0,
        'init-checkpoint':str(checkpoint), 'output':str(output/'fit'),
        'radiance-residual':True, 'residual-normal':normal_source,
        'residual-interaction':interaction, 'residual-angular-bank':angular_bank,
        'residual-paired-context':paired_context, 'residual-pair-weight':pair_weight,
        'residual-rays':2048, 'residual-peak-fraction':.25, 'residual-context-fraction':.25,
        'normal-weight':0., 'depth-weight':0., 'refine-stop':0, 'validate-every':0})
    if residual_frames is not None:
        options['residual-frames'] = residual_frames
    sample_pool = source['fit_indices'] if residual_frames is None else residual_frames
    schedule_rng = random.Random(0)
    expected_schedule = [schedule_rng.choice(sample_pool) for _ in range(3)]
    # The CLI child owns the GPU during training. Keep only CPU reference state.
    del g, model, saved, sample, offsets, parameter
    gc.collect()
    torch.cuda.empty_cache()
    previous_head = None
    initial_head = None
    interaction_keys = ('interaction_spatial.weight', 'interaction_angular.weight',
                        'interaction_projection.weight')
    for phase, expected_steps in [('fit', 3), ('resume', 6)]:
        command = train_argv(sys.executable, options)
        (output/f'{phase}.command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (output/f'{phase}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        state = torch.load(output/phase/'last.pt', map_location='cpu', weights_only=False)
        assert state['residual_steps'] == expected_steps
        assert state['config']['residual_frames'] == residual_frames
        assert state['residual_sample_indices'] == sample_pool
        expected_counts = {index:expected_schedule.count(index) for index in sample_pool}
        assert state['residual_frame_counts'] == expected_counts
        assert sum(state['residual_frame_counts'].values()) == 3
        assert state['config']['residual_normal'] == normal_source
        assert state['config']['residual_interaction'] == interaction
        assert state['config']['residual_angular_bank'] == angular_bank
        assert state['config']['residual_paired_context'] == paired_context
        assert state['config']['residual_pair_weight'] == pair_weight
        assert state['config']['surface_depth'] == cfg['surface_depth']
        for group in ('gaussians', 'transport', 'camera_offsets'):
            assert state[group].keys() == source[group].keys()
            for key, value in source[group].items():
                torch.testing.assert_close(state[group][key], value, rtol=0, atol=0)
        assert state['radius'] == source['radius']
        assert state['fit_indices'] == source['fit_indices'] and state['val_indices'] == source['val_indices']
        assert all(torch.isfinite(value).all() for value in state['radiance_residual'].values())
        if phase == 'fit':
            initial = torch.load(output/phase/'residual_initial.pt', map_location='cpu', weights_only=False)
            assert initial['residual_steps'] == 0
            assert initial['config']['residual_interaction'] == interaction
            assert initial['config']['residual_angular_bank'] == angular_bank
            assert initial['config']['residual_normal'] == normal_source
            initial_head = initial['radiance_residual']
            assert initial_head.keys() == state['radiance_residual'].keys()
            assert torch.count_nonzero(initial_head['network.6.weight']) == 0
            assert torch.count_nonzero(initial_head['network.6.bias']) == 0
        else:
            assert not (output/phase/'residual_initial.pt').exists()
        if interaction != 'none':
            prior = initial_head if phase == 'fit' else previous_head
            assert all(not torch.equal(state['radiance_residual'][key], prior[key])
                       for key in interaction_keys)
        if angular_bank != 'none':
            prior = initial_head if phase == 'fit' else previous_head
            keys = [key for key in state['radiance_residual']
                    if key.startswith(('center_network.','angular_projection.'))]
            assert all(not torch.equal(state['radiance_residual'][key],prior[key]) for key in keys)
            assert torch.equal(state['radiance_residual']['angular_scales'],initial_head['angular_scales'])
        if previous_head is None:
            assert state['radiance_residual']['network.6.weight'].abs().sum()>0
        else:
            assert any(not torch.equal(value, previous_head[key])
                       for key,value in state['radiance_residual'].items())
        previous_head = state['radiance_residual']
        head = load_radiance_residual(state)
        for key,value in head.state_dict().items():
            torch.testing.assert_close(value, state['radiance_residual'][key], rtol=0, atol=0)
        try:
            load_radiance_residual({k:v for k,v in state.items() if k!='radiance_residual'})
        except ValueError:
            pass
        else:
            raise AssertionError('Missing enabled residual weights were accepted')
        if interaction != 'none':
            for missing_mode in (False, True):
                invalid = dict(state, config=dict(state['config']))
                if missing_mode:
                    del invalid['config']['residual_interaction']
                else:
                    invalid['config']['residual_interaction'] = 'none'
                try:
                    load_radiance_residual(invalid)
                except (ValueError, RuntimeError):
                    pass
                else:
                    raise AssertionError('Interaction weights accepted without their enabled mode')
        if angular_bank != 'none':
            for invalid_mode in ('missing','none','narrow' if angular_bank == 'wide' else 'wide'):
                invalid = dict(state,config=dict(state['config']))
                if invalid_mode == 'missing':
                    del invalid['config']['residual_angular_bank']
                else:
                    invalid['config']['residual_angular_bank'] = invalid_mode
                try:
                    load_radiance_residual(invalid)
                except (ValueError,RuntimeError):
                    pass
                else:
                    raise AssertionError('Angular state accepted with a missing or incompatible bank')
        history = [json.loads(line) for line in (output/phase/'history.jsonl').read_text().splitlines()]
        if paired_context:
            eligibility_events = [row for row in history if row.get('event') == 'residual_pair_eligibility']
            assert {row['frame_index'] for row in eligibility_events} == set(expected_schedule)
            history = [row for row in history if 'stage' in row]
        assert all(row['stage']=='gaussian_residual' for row in history)
        expected_loss_terms = {'residual_rgb', 'residual_pair'} if paired_context else {'residual_rgb'}
        assert all(set(row['loss_terms']) == expected_loss_terms for row in history)
        if paired_context:
            for row in history:
                stats = row['residual_pair_stats']
                assert stats['nominal_pairs'] == 512
                assert 0 <= stats['supported_pairs'] <= stats['sampled_pairs'] <= 512
                assert stats['excluded_pairs'] == stats['sampled_pairs']-stats['supported_pairs']
                assert math.isfinite(stats['pair_loss']) and stats['pair_loss'] >= 0
                assert math.isclose(stats['weighted_pair_loss'], pair_weight*stats['pair_loss'],
                                    rel_tol=1e-6, abs_tol=1e-8)
                assert stats['weighted_pair_loss'] == row['loss_terms']['residual_pair']
                assert math.isclose(row['loss'], sum(row['loss_terms'].values()), rel_tol=1e-6, abs_tol=1e-8)
            assert history[-1]['residual_pair_stats']['sampled_pairs'] == 512
            assert history[-1]['residual_pair_stats']['supported_pairs'] > 0
            assert state['residual_pair_eligibility']
            totals = state['residual_pair_totals']
            assert totals['draws'] == 3*2048 and totals['nominal_pairs'] == 3*512
            assert totals['sampled_pairs'] == totals['supported_pairs']+totals['excluded_pairs']
            assert totals['sampled_pairs'] == totals['pair_peak_draws'] == totals['pair_ring_draws']
            assert totals['draws'] == 2*totals['sampled_pairs']+totals['foreground_draws']+totals['valid_draws']
            assert totals == history[-1]['residual_pair_totals']
            audit = json.loads((output/phase/'residual_pair_audit.json').read_text())
            assert audit['stage_steps'] == 3 and audit['residual_pair_totals'] == totals
            report[phase+'_paired_loss'] = {'nominal_denominator':512, 'logged_loss_contract':True,
                'configuration_and_sampler_persisted':True, 'stage_totals':state['residual_pair_totals']}
        for row in history:
            assert row['frame_index'] == expected_schedule[row['step']-1]
            prefix_counts = {str(index):expected_schedule[:row['step']].count(index) for index in sample_pool}
            assert row['residual_frame_counts'] == prefix_counts
        assert history[-1]['residual_steps'] == expected_steps
        assert history[-1]['residual_rays_total'] == 3*2048
        assert history[-1]['residual_peak_rays_total']>0 and history[-1]['residual_context_rays_total']>0
        assert math.isclose(history[-1]['residual_lr'], .00028)
        assert history[-1]['residual_stats']['requested_pixels'] == 2048
        if interaction != 'none':
            optimization = history[-1]['residual_optimization_stats']
            assert all(math.isfinite(value) for value in optimization.values())
            for row in history:
                for name, value in row['residual_stats'].items():
                    if name.startswith('interaction_'):
                        assert math.isfinite(value) and value >= 0
            for name in ('interaction_spatial', 'interaction_angular', 'interaction_projection'):
                assert optimization[name+'_gradient_rms'] > 0
                assert optimization[name+'_parameter_rms'] > 0
        if angular_bank != 'none':
            if phase == 'fit':
                first = history[0]
                assert first['step'] == 1 and first['residual_stats']['center_offset_square_mean'] == 0
                assert all(first['residual_optimization_stats'][name+'_gradient_rms'] == 0
                           for name in ('angular_center_first','angular_center_last','angular_projection'))
            optimization = history[-1]['residual_optimization_stats']
            for name in ('angular_center_first','angular_center_last','angular_projection'):
                assert optimization[name+'_gradient_rms'] > 0
            for row in history:
                assert all(math.isfinite(value) for value in row['residual_stats'].values())
                for j in range(8):
                    assert optimization[f'angular_kernel_{j}_projection_gradient_rms'] >= 0
            assert history[-1]['residual_stats']['center_offset_square_mean'] > 0
            assert history[-1]['residual_stats']['center_rotation_deg_mean'] > 0
            assert history[-1]['residual_stats']['center_raw_norm_min'] > 0
        report[phase] = {'residual_steps':expected_steps, 'parent_and_camera_state_bitwise_unchanged':True,
            'head_changed':True, 'strict_reload':True, 'missing_weights_rejected':True,
            'residual_sample_indices':sample_pool, 'residual_frame_counts':expected_counts,
            'source_split_preserved':True, 'stage_counts_restart_on_resume':True,
            'final_history':history[-1]}
        if interaction != 'none':
            report[phase].update(interaction_weights_updated=True,
                interaction_gradients_nonzero=True, disabled_or_missing_mode_rejected=True)
        if angular_bank != 'none':
            report[phase].update(angular_center_and_projection_updated=True,
                angular_center_gradients_nonzero=True,angular_bank_load_guards=True)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        options.update({'init-checkpoint':str(output/phase/'last.pt'), 'output':str(output/'resume')})
        del state, head
        print(f'PASS: residual {phase}, {expected_steps} accumulated steps, exact frozen parent state', flush=True)
    if interaction != 'none' and angular_bank == 'none':
        wrong = dict(options, **{'residual-interaction':'multiply' if interaction == 'add' else 'add',
                                'output':str(output/'wrong_mode')})
        command = train_argv(sys.executable, wrong)
        (output/'wrong_mode.command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (output/'wrong_mode.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode != 0
        failure = (output/'wrong_mode.log').read_text()
        assert 'Resuming a radiance residual requires its saved --residual-interaction' in failure
        assert not (output/'wrong_mode/history.jsonl').exists()
        report['wrong_mode_resume'] = {'exit_code':result.returncode,
                                     'rejected_before_training':True}
    if angular_bank != 'none':
        wrong = dict(options, **{'residual-angular-bank':'narrow' if angular_bank == 'wide' else 'wide',
                                'output':str(output/'wrong_bank')})
        command = train_argv(sys.executable,wrong)
        (output/'wrong_bank.command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (output/'wrong_bank.log').open('w') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode != 0
        assert 'Resuming a radiance residual requires its saved --residual-angular-bank' in (output/'wrong_bank.log').read_text()
        assert not (output/'wrong_bank/history.jsonl').exists()
        report['wrong_bank_resume'] = {'exit_code':result.returncode,'rejected_before_training':True}
    if paired_context:
        for label, changes, flag in (
                ('wrong_pair_mode', {'residual-paired-context':False, 'residual-pair-weight':0.},
                 '--residual-paired-context'),
                ('wrong_pair_weight', {'residual-pair-weight':.25 if pair_weight == 0 else 0.},
                 '--residual-pair-weight')):
            wrong = dict(options, **changes, output=str(output/label))
            command = train_argv(sys.executable, wrong)
            (output/f'{label}.command.json').write_text(json.dumps(command, indent=2)+'\n')
            with (output/f'{label}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            assert result.returncode != 0
            assert f'Resuming a radiance residual requires its saved {flag}' in (output/f'{label}.log').read_text()
            assert not (output/label/'history.jsonl').exists()
            report[label] = {'exit_code':result.returncode, 'rejected_before_training':True}
    # Reconstruct the saved head twice and compare actual receiver rendering,
    # including the saved fit camera, rather than only comparing state tensors.
    g, model, saved = load_model(output/'resume/last.pt')
    sample = to_device(SceneDataset(cfg['scene'], 'train', 128,
        unit_light_intensity=cfg['unit_light_intensity'])[source['fit_indices'][1]], 'cuda')
    offsets = None
    if saved['camera_offsets'] is not None:
        offsets = TrainCameraOffsets(len(saved['fit_indices']), g.radius).cuda()
        offsets.load_state_dict(saved['camera_offsets'])
        sample = offsets.correct(sample, 1)
    head = load_radiance_residual(saved)
    reloaded_head = load_radiance_residual(torch.load(output/'resume/last.pt', map_location='cpu', weights_only=False)).cuda()
    with torch.no_grad():
        rendered = render_observation(g, model, sample, cfg['background'], shadow, port_active,
            cfg['display_gamma'], cfg['shadow_mode'], radiance_residual=head)[0]
        restored = render_observation(g, model, sample, cfg['background'], shadow, port_active,
            cfg['display_gamma'], cfg['shadow_mode'], radiance_residual=reloaded_head)[0]
    torch.testing.assert_close(rendered, restored, rtol=0, atol=0)
    assert torch.isfinite(rendered).all()
    report['saved_head_render_bitwise_reloaded'] = True
    del g, model, saved, sample, offsets, head, reloaded_head, rendered, restored
    gc.collect()
    torch.cuda.empty_cache()
    command = [sys.executable, str(Path(__file__).parent/'evaluate.py'), str(output/'resume/last.pt'),
               '--split', 'fit', '--limit', '1', '--resolution', '512', '--highlights',
               '--output', str(output/'evaluation')]
    (output/'evaluation.command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (output/'evaluation.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    evaluation = json.loads((output/'evaluation/metrics.json').read_text())
    assert evaluation['render_branch']=='gaussian_residual'
    assert evaluation['corrected_fit_frames']==1 and evaluation['evaluated_frames']==1
    assert evaluation['resolution']==512 and 'neutral_peak_metrics' in evaluation['metrics']
    report['evaluation'] = evaluation
    report['complete'] = True
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: native residual CLI evaluation with saved fit camera and highlight metrics', flush=True)


def check_material_checkpoint(checkpoint, output, optimize_light_scale=False):
    """Convert a real neural-material fit to GGX, train, resume and render."""
    from make_validation_manifest import train_argv
    from materials.ggx import GGXMaterial
    torch.set_num_threads(8)
    output.mkdir(parents=True, exist_ok=False)
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = resolve_config(source['config'])
    keys = set(METHODS['neural_material'].cli_fields) | {
        'scene', 'representation', 'resolution', 'background', 'display_gamma',
        'shadow_mode', 'max_points', 'fit_all', 'surface_depth'}
    options = {key.replace('_', '-'):cfg[key] for key in keys}
    options.update({'steps':3, 'init-checkpoint':str(checkpoint), 'output':str(output/'fit'),
                    'material-model':'ggx', 'reset-material':True, 'freeze-geometry':True,
                    'normal-weight':0, 'depth-weight':0, 'refine-stop':0,
                    'shadow-start':1, 'port-start':1, 'validate-every':0})
    if source['camera_offsets'] is not None:
        options.update({'optimize-cameras':True, 'camera-start':1, 'camera-lr':0.})
    if optimize_light_scale:
        options['optimize-light-scale'] = True
    for phase in ('fit', 'resume'):
        command = train_argv(sys.executable, options)
        (output/f'{phase}.command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (output/f'{phase}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        g, model, state = load_model(output/phase/'last.pt')
        assert isinstance(model.decoder, GGXMaterial)
        assert not any(key.startswith('decoder.') for key in state['transport'])
        for key in ['means', 'scales', 'quats', 'opacities']:
            torch.testing.assert_close(state['gaussians']['params.'+key].cpu(), source['gaussians']['params.'+key], rtol=0, atol=0)
        if source['camera_offsets'] is not None:
            for key, value in source['camera_offsets'].items():
                torch.testing.assert_close(state['camera_offsets'][key].cpu(), value, rtol=0, atol=0)
        assert all(torch.isfinite(t).all() for t in state['gaussians'].values())
        assert all(torch.isfinite(t).all() for t in state['transport'].values())
        if optimize_light_scale:
            scale = state['transport']['light_scale'].cpu()
            assert float(scale)>0 and not torch.equal(scale, source['transport']['light_scale'])
        if phase == 'fit':
            initial = model.decoder.initial_latent(g.params['features'])
            assert (g.params['features'][:, :6].sigmoid()-initial).abs().max()>0
        sample = to_device(SceneDataset(cfg['scene'], 'train', cfg['resolution'])[state['fit_indices'][0]], 'cuda')
        with torch.no_grad():
            rendered = render_observation(g, model, sample, cfg['background'], True, True,
                                          cfg['display_gamma'], cfg['shadow_mode'])[0]
            restored_g, restored_model, _ = load_model(output/phase/'last.pt')
            reloaded = render_observation(restored_g, restored_model, sample, cfg['background'], True, True,
                                          cfg['display_gamma'], cfg['shadow_mode'])[0]
        torch.testing.assert_close(rendered, reloaded, rtol=0, atol=0)
        assert torch.isfinite(rendered).all()
        options.update({'init-checkpoint':str(output/phase/'last.pt'), 'output':str(output/'resume'), 'reset-material':False})
        del g, model, state, sample, rendered, reloaded, restored_g, restored_model
        torch.cuda.empty_cache()
    with (output/'evaluation.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).parent/'evaluate.py'),
                        str(output/'resume/last.pt'), '--split', 'fit', '--limit', '1',
                        '--output', str(output/'evaluation')], stdout=log, stderr=subprocess.STDOUT, check=True)
    (output/'report.json').write_text(json.dumps({'material_model':'ggx', 'fit_and_resume':True,
        'geometry_unchanged':True, 'camera_offsets_unchanged':True,
        'finite':True, 'bitwise_render_reload':True,
        'positive_fitted_light_scale':optimize_light_scale}, indent=2)+'\n')
    print('PASS: real material conversion, fitting, resume and exact render reload', flush=True)


def check_sdf_checkpoint(checkpoint, output, shading=False, primitive_weight=0., freeze_sdf=False,
                         normal_field=False, surface_depth='center', sdf_start=1, volume=False, volume_only=False,
                         sdf_detail=False, volume_detail=False, volume_peak_fraction=0., resolution=None,
                         volume_context_fraction=0., volume_hint_encoding=False, volume_rays=None,
                         volume_samples=None, volume_fixed_sharpness=None):
    """Exercise mutual gradients and CLI save/resume on a real fitted surface."""
    from sdf import SurfaceSDF, sample_surface, visible_primitives
    from make_validation_manifest import train_argv
    torch.set_num_threads(8)
    output.mkdir(parents=True, exist_ok=False)
    g, model, saved = load_model(checkpoint)
    g.surface_depth = surface_depth
    cfg = resolve_config(saved['config'])
    if resolution is not None:
        cfg['resolution'] = resolution
    sdf_detail = sdf_detail or cfg.get('sdf_detail', False)
    volume_detail = volume_detail or cfg.get('sdf_volume_detail', False)
    volume_hint_encoding = volume_hint_encoding or cfg.get('sdf_volume_hint_encoding', False)
    initial_steps = saved.get('sdf_steps', 0)
    initial_field = saved['sdf'] if freeze_sdf else None
    initial_cameras = saved['camera_offsets']
    initial_volume_steps = saved.get('sdf_volume_steps', 0)
    previous_volume = saved.get('sdf_volume') if volume_fixed_sharpness is not None else None
    if volume_only:
        initial_gaussians, initial_transport = saved['gaussians'], saved['transport']
    sample = to_device(SceneDataset(cfg['scene'], 'train', cfg['resolution'])[saved['fit_indices'][0]], 'cuda')
    field = SurfaceSDF(g.center, g.radius, detail=sdf_detail)
    if volume and (sdf_detail or volume_detail or volume_hint_encoding):
        from sdf import silhouette_rays
        from sdf_volume import SDFRadiance, render_rays
        old_field, old_head, ray_samples = load_sdf_volume(saved)
        if volume_samples is not None:
            ray_samples = volume_samples
        expanded_field = SurfaceSDF(g.center, g.radius, detail=sdf_detail)
        expanded_head = SDFRadiance('cuda', detail=volume_detail, hint_encoding=volume_hint_encoding)
        expanded_field.load_state_dict({**expanded_field.state_dict(), **old_field.state_dict()})
        expanded_head.load_state_dict({**expanded_head.state_dict(), **old_head.state_dict()})
        if volume_fixed_sharpness is not None:
            # Test zero-residual expansion at the same chosen sharpness;
            # changing sharpness need not preserve the source rendering.
            with torch.no_grad():
                old_head.log_sharpness.fill_(math.log(volume_fixed_sharpness))
                expanded_head.log_sharpness.fill_(math.log(volume_fixed_sharpness))
        rays = silhouette_rays(sample, old_field)
        ids = rays['valid'][::max(1, len(rays['valid'])//64)][:64]
        with torch.no_grad():
            original = render_rays(old_field, old_head, sample, rays, ids, model.light_scale, samples=ray_samples)
            expanded = render_rays(expanded_field, expanded_head, sample, rays, ids, model.light_scale, samples=ray_samples)
        for key in ['linear', 'alpha', 'depth', 'normal']:
            torch.testing.assert_close(original[key], expanded[key], rtol=0, atol=0)
        del old_field, old_head, expanded_field, expanded_head, rays, original, expanded
        torch.cuda.empty_cache()
    _, alpha, info = render_observation(g, model, sample, cfg['background'], True, True,
                                         cfg['display_gamma'], cfg['shadow_mode'])
    generator = torch.Generator(device='cuda').manual_seed(1)
    points, normals, toward = sample_surface(info, alpha, sample, field, 128, generator)
    sum(field.fit_losses(points, normals, toward).values()).backward()
    assert all(p.grad is None for p in g.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in field.parameters())
    field.zero_grad(set_to_none=True)
    sum(field.geometry_losses(points, normals, .05, .01).values()).backward()
    assert all(p.grad is None for p in field.parameters())
    for key in ('means', 'quats'):
        grad = g.params[key].grad
        assert torch.isfinite(grad).all() and grad.abs().sum()>0, key
    if primitive_weight:
        g.zero_grad(set_to_none=True)
        field.zero_grad(set_to_none=True)
        selected = visible_primitives(g.params['means'], g.params['opacities'].sigmoid(),
                                      info, alpha, sample, g.radius)
        assert len(selected)>0
        field.primitive_loss(g.params['means'][selected[:1024]]).backward()
        assert torch.isfinite(g.params['means'].grad).all() and g.params['means'].grad.abs().sum()>0
        assert g.params['opacities'].grad is None
        assert all(p.grad is None for p in field.parameters())
    if shading:
        field = load_surface_field(saved)
        g.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        image, _, _ = render_observation(g, model, sample, cfg['background'], True, True,
                                         cfg['display_gamma'], cfg['shadow_mode'], surface_field=field)
        target = target_image(sample, cfg['background'], cfg['display_gamma'])
        (image-target).square().mean().backward()
        gradients = [p.grad for p in field.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        assert sum(float(grad.abs().sum()) for grad in gradients)>0
        assert torch.isfinite(g.params['means'].grad).all()
        assert all(p.grad is None for p in model.decoder.parameters())
        del image, target
    if normal_field:
        from materials.normal_field import NormalResidualField
        detail = NormalResidualField(g.center, g.radius)
        g.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            reference = render_observation(g, model, sample, cfg['background'], True, True,
                                            cfg['display_gamma'], cfg['shadow_mode'])[0]
        image, _, _ = render_observation(g, model, sample, cfg['background'], True, True,
                                         cfg['display_gamma'], cfg['shadow_mode'], normal_field=detail)
        torch.testing.assert_close(image.detach(), reference, rtol=0, atol=0)
        (image-target_image(sample,cfg['background'],cfg['display_gamma'])).square().mean().backward()
        grads = [p.grad for p in detail.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert sum(float(grad.abs().sum()) for grad in grads)>0
        assert all(p.grad is None for p in model.decoder.parameters())
        del detail, reference, image
    keys = set(METHODS[cfg['representation']].cli_fields) | {
        'scene', 'representation', 'resolution', 'feature_dim', 'background',
        'display_gamma', 'shadow_mode', 'max_points', 'fit_all'}
    options = {key.replace('_', '-'):cfg[key] for key in keys}
    if resolution is not None:
        options['resolution'] = resolution
    options.update({'steps':3, 'init-checkpoint':str(checkpoint), 'output':str(output/'fit'),
                    'sdf':True, 'sdf-warmup-steps':0, 'sdf-samples':128,
                    'surface-depth':surface_depth,
                    'sdf-start':sdf_start,
                    'normal-weight':0, 'depth-weight':0, 'refine-stop':0,
                    'shadow-start':1, 'port-start':1, 'validate-every':0})
    if shading:
        options['sdf-shading'] = True
    if primitive_weight:
        options['sdf-primitive-weight'] = primitive_weight
    if freeze_sdf:
        options['freeze-sdf'] = True
    if normal_field:
        options['normal-field'] = True
    if initial_cameras is not None and not volume_only:
        options.update({'optimize-cameras':True, 'camera-start':1, 'camera-lr':0.})
    if volume:
        options.update({'sdf-volume-weight':.1, 'sdf-volume-rays':512 if sdf_detail or volume_detail else 16,
                        'sdf-volume-samples':64 if sdf_detail or volume_detail else 16, 'sdf-volume-warmup':2,
                        'sdf-volume-detail':volume_detail, 'sdf-volume-peak-fraction':volume_peak_fraction,
                        'sdf-volume-peak-context-fraction':volume_context_fraction,
                        'sdf-volume-hint-encoding':volume_hint_encoding})
        if volume_rays is not None:
            options['sdf-volume-rays'] = volume_rays
        if volume_samples is not None:
            options['sdf-volume-samples'] = volume_samples
        if volume_fixed_sharpness is not None:
            options['sdf-volume-fixed-sharpness'] = volume_fixed_sharpness
    options.update({'sdf-detail':sdf_detail, 'sdf-lr':cfg.get('sdf_lr', .001)})
    if volume_only:
        options.update({'sdf-volume-only':True, 'camera-start':1})
    del g, model, saved, sample, field, points, normals, toward, alpha, info
    torch.cuda.empty_cache()
    for phase in ('fit', 'resume'):
        command = train_argv(sys.executable, options)
        (output/f'{phase}.command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (output/f'{phase}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        g, model, state = load_model(output/phase/'last.pt')
        assert g.surface_depth == surface_depth
        if initial_cameras is not None:
            for key, value in initial_cameras.items():
                torch.testing.assert_close(state['camera_offsets'][key], value, rtol=0, atol=0)
        updates = max(0, 4-sdf_start)*(1 if phase=='fit' else 2)
        expected_steps = initial_steps if freeze_sdf else initial_steps+updates
        if volume_only and not freeze_sdf:
            expected_steps = initial_steps+max(0, initial_volume_steps+updates-2)-max(0, initial_volume_steps-2)
        assert state['sdf_steps'] == expected_steps
        if volume_only:
            for before, after in [(initial_gaussians, state['gaussians']), (initial_transport, state['transport'])]:
                for key, value in before.items():
                    torch.testing.assert_close(after[key], value, rtol=0, atol=0)
        history = [json.loads(line) for line in (output/phase/'history.jsonl').read_text().splitlines()]
        if sdf_start > 1:
            previous_updates = updates//2 if phase=='resume' and not freeze_sdf else 0
            assert history[0]['sdf_steps'] == initial_steps+previous_updates
            assert not any(k.startswith('sdf_') for k in history[0]['loss_terms'])
        if freeze_sdf:
            for key, value in initial_field.items():
                torch.testing.assert_close(state['sdf'][key], value, rtol=0, atol=0)
        assert all(torch.isfinite(t).all() for t in state['sdf'].values())
        if volume:
            from sdf import silhouette_rays
            from sdf_volume import render_rays
            assert state['sdf_volume_steps'] == initial_volume_steps+updates
            assert all(torch.isfinite(t).all() for t in state['sdf_volume'].values())
            assert history[-1]['loss_terms']['sdf_volume_rgb']>0
            assert 'sdf_ray_depth_to_field' in history[-1]['loss_terms']
            assert 'sdf_fit' not in history[-1]['loss_terms']
            if volume_fixed_sharpness is not None:
                expected_log = state['sdf_volume']['log_sharpness'].new_tensor(math.log(volume_fixed_sharpness))
                torch.testing.assert_close(state['sdf_volume']['log_sharpness'], expected_log, rtol=0, atol=0)
                assert state['config']['sdf_volume_fixed_sharpness'] == volume_fixed_sharpness
                sharpness_history = [entry['sdf_volume_sharpness'] for entry in history
                                     if 'sdf_volume_sharpness' in entry]
                assert sharpness_history and all(value == float(expected_log.exp()) for value in sharpness_history)
                if previous_volume is not None:
                    assert any(not torch.equal(value, previous_volume[key])
                               for key,value in state['sdf_volume'].items() if key.startswith('network.'))
                previous_volume = state['sdf_volume']
            sample = to_device(SceneDataset(cfg['scene'], 'train', cfg['resolution'])[state['fit_indices'][0]], 'cuda')
            first_field, first_head, samples = load_sdf_volume(state)
            if volume_fixed_sharpness is not None:
                torch.testing.assert_close(first_head.log_sharpness, expected_log, rtol=0, atol=0)
            rays = silhouette_rays(sample, first_field)
            ids = rays['valid'][::max(1, len(rays['valid'])//64)][:64]
            with torch.no_grad():
                first = render_rays(first_field, first_head, sample, rays, ids, model.light_scale, samples=samples)
                second_field, second_head, _ = load_sdf_volume(state)
                second = render_rays(second_field, second_head, sample, rays, ids, model.light_scale, samples=samples)
            for key in ['linear', 'alpha', 'depth', 'normal']:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
                assert torch.isfinite(first[key]).all()
            del sample, first_field, first_head, second_field, second_head, first, second, rays
        if primitive_weight:
            history = [json.loads(line) for line in (output/phase/'history.jsonl').read_text().splitlines()]
            assert history[-1]['loss_terms']['sdf_primitive']>0
        field = load_surface_field(state)
        if normal_field:
            detail = load_normal_field(state)
            sample = to_device(SceneDataset(cfg['scene'], 'train', cfg['resolution'])[state['fit_indices'][0]], 'cuda')
            with torch.no_grad():
                rendered = render_observation(g, model, sample, cfg['background'], True, True,
                    cfg['display_gamma'], cfg['shadow_mode'], normal_field=detail)[0]
                restored = load_normal_field(state)
                reloaded = render_observation(g, model, sample, cfg['background'], True, True,
                    cfg['display_gamma'], cfg['shadow_mode'], normal_field=restored)[0]
            torch.testing.assert_close(rendered, reloaded, rtol=0, atol=0)
            assert torch.isfinite(rendered).all()
            assert detail.network[-1].weight.abs().sum()>0
            del detail, restored, rendered, reloaded, sample
        if shading:
            sample = to_device(SceneDataset(cfg['scene'], 'train', cfg['resolution'])[state['fit_indices'][0]], 'cuda')
            with torch.no_grad():
                rendered = render_observation(g, model, sample, cfg['background'], True, True,
                                              cfg['display_gamma'], cfg['shadow_mode'], surface_field=field)[0]
                restored = load_surface_field(state)
                reloaded = render_observation(g, model, sample, cfg['background'], True, True,
                                              cfg['display_gamma'], cfg['shadow_mode'], surface_field=restored)[0]
                gaussian_normal = render_observation(g, model, sample, cfg['background'], True, True,
                                                     cfg['display_gamma'], cfg['shadow_mode'])[0]
            torch.testing.assert_close(rendered, reloaded, rtol=0, atol=0)
            assert float((rendered-gaussian_normal).abs().max())>1e-4
            assert torch.isfinite(rendered).all()
            del sample, rendered, reloaded, gaussian_normal, restored
        options.update({'init-checkpoint':str(output/phase/'last.pt'), 'output':str(output/'resume')})
        del g, model, state, field
        torch.cuda.empty_cache()
    (output/'report.json').write_text(json.dumps({'mutual_geometry_gradients':True,
        'sdf_steps_after_reload':expected_steps, 'finite_field':True, 'sdf_shading':shading,
        'sdf_primitive_weight':primitive_weight, 'freeze_sdf':freeze_sdf,
        'normal_field':normal_field, 'surface_depth':surface_depth,
        'sdf_start':sdf_start, 'preserved_fitted_cameras':initial_cameras is not None}, indent=2)+'\n')
    if shading or normal_field or surface_depth == 'intersection' or volume:
        with (output/'evaluation.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).parent/'evaluate.py'),
                            str(output/'resume/last.pt'), '--split', 'fit', '--limit', '1',
                            '--output', str(output/'evaluation')], stdout=log, stderr=subprocess.STDOUT, check=True)
    if volume:
        with (output/'volume_evaluation.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).parent/'evaluate.py'),
                            str(output/'resume/last.pt'), '--split', 'fit', '--limit', '1', '--sdf-volume',
                            '--output', str(output/'volume_evaluation')], stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output/'report.json').read_text())
        report.update(sdf_volume=True, sdf_volume_steps=initial_volume_steps+updates,
                      volume_bitwise_reload=True, volume_state_finite=True, fixed_gaussian_teacher=volume_only,
                      sdf_detail=sdf_detail, volume_detail=volume_detail,
                      volume_hint_encoding=volume_hint_encoding, training_resolution=cfg['resolution'],
                      sdf_volume_peak_fraction=volume_peak_fraction, sdf_volume_peak_context_fraction=volume_context_fraction,
                      sdf_volume_rays=options['sdf-volume-rays'],
                      sdf_volume_samples=options['sdf-volume-samples'],
                      sdf_volume_fixed_sharpness=volume_fixed_sharpness,
                      fixed_sharpness_history_and_reload=volume_fixed_sharpness is not None,
                      detail_initial_render_preserved=bool(sdf_detail or volume_detail or volume_hint_encoding)
                                                     and volume_fixed_sharpness is None,
                      detail_render_preserved_at_fixed_sharpness=bool(sdf_detail or volume_detail or volume_hint_encoding)
                                                               and volume_fixed_sharpness is not None)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: real 2DGS/SDF mutual gradients, CLI fit, checkpoint reload and resumed field', flush=True)


def check_geometry_warmup(g, model, sample):
    optimizers = g.optimizers()
    network_optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    original_network = {k: p.detach().clone() for k, p in model.named_parameters()}
    original_features = g.params['features'].detach().clone()
    original_means = g.params['means'].detach().clone()
    for step in (1, 2):
        assert set_training_stage(g, model, optimizers, step, 2)
        network_optimizer.zero_grad(set_to_none=True)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        image, alpha, _ = render_observation(g, model, sample, 0., True, True, 2.2, 'deep', geometry_only=True)
        changed_light = dict(sample, light_intensity=sample['light_intensity']*5,
                             light_pos=sample['light_pos']+2)
        other, _, _ = render_observation(g, model, changed_light, 0., True, True, 2.2, 'deep', geometry_only=True)
        torch.testing.assert_close(image, other, rtol=0, atol=0)
        (image.square().mean()+alpha.mean()).backward()
        assert all(p.grad is None for p in model.parameters())
        assert g.params['features'].grad is None
        for optimizer in optimizers.values():
            optimizer.step()
    assert all(torch.equal(p, original_network[k]) for k,p in model.named_parameters())
    assert torch.equal(g.params['features'], original_features)
    assert not torch.equal(g.params['means'], original_means)
    trained_geometry = g.params['means'].detach().clone()
    assert not set_training_stage(g, model, optimizers, 3, 2)
    torch.testing.assert_close(g.params['means'], trained_geometry, rtol=0, atol=0)
    assert bool((g.params['base']==-1.5).all()) and not optimizers['base'].state
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    network_optimizer.zero_grad(set_to_none=True)
    image, _, _ = render_observation(g, model, sample, 0., True, True, 2.2, 'deep')
    image.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    network_optimizer.step()
    assert any(not torch.equal(p, original_network[k]) for k,p in model.named_parameters())
    print('PASS: warmup ignores lighting, freezes transport/features, updates geometry; transition resets RGB and trains transport', flush=True)


def check_surfel_shadow():
    g = Gaussians(2, torch.tensor([0., 0., 3.], device='cuda'), 1., geometry='2dgs')
    with torch.no_grad():
        g.params['means'].copy_(torch.tensor([[0., 0., 2.], [0., 0., 4.]], device='cuda'))
        g.params['quats'].zero_(); g.params['quats'][:, 0] = 1.
        g.params['scales'].fill_(torch.tensor(.3).log())
        g.params['opacities'].fill_(torch.logit(torch.tensor(.7)))
    visibility = visibility_hint(g, torch.zeros(3, device='cuda'), mode='deep')
    assert visibility[0] > .99, visibility  # No self-shadow for the front disk.
    assert .25 < visibility[1] < .4, visibility  # Approximately 1 - foreground opacity.
    visibility[1].backward()
    assert torch.isfinite(g.params['opacities'].grad).all()
    assert g.params['opacities'].grad[0] < 0
    return visibility.detach().tolist()


def check_surfel_channel_padding():
    """Discarded feature channels must not inject allocator contents into gradients."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        inputs = {
            'means': torch.tensor([[0., 0., 3.]], device='cuda', requires_grad=True),
            'quats': torch.tensor([[1., 0., 0., 0.]], device='cuda', requires_grad=True),
            'scales': torch.tensor([[.2, .2, 1.]], device='cuda', requires_grad=True),
            'opacities': torch.tensor([.7], device='cuda', requires_grad=True),
            'colors': torch.full((1, 36), .2, device='cuda', requires_grad=True),
            'viewmats': torch.eye(4, device='cuda')[None],
            'Ks': torch.tensor([[[50., 0., 32.], [0., 50., 32.], [0., 0., 1.]]], device='cuda'),
            'width': 64, 'height': 64, 'packed': False, 'absgrad': True,
            'render_mode': 'RGB+ED', 'backgrounds': torch.zeros(1, 36, device='cuda'),
        }
        rendered, alpha, info = _rasterize('2dgs', **inputs)
        with torch.no_grad():
            reference = rasterization_2dgs(**dict(inputs, colors=inputs['colors'][None]), distloss=True)
        assert rendered.shape[-1] == 37
        torch.testing.assert_close(rendered, reference[0], rtol=0, atol=0)
        torch.testing.assert_close(alpha, reference[1], rtol=0, atol=0)
        (rendered.mean()+info['surface_normals'].square().mean()+info['surface_distortion'].mean()).backward()
        for name in ('means', 'quats', 'scales', 'opacities', 'colors'):
            assert torch.isfinite(inputs[name].grad).all(), name
        print('PASS: 2DGS feature padding preserves forward output and finite gradients with poisoned allocations', flush=True)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill


def check_surfel_depth():
    """A tilted planar surfel must reconstruct its own plane, including gradients."""
    import math
    c, s = math.cos(.4), math.sin(.4)
    view = torch.tensor([[c, 0, s, .2], [0, 1, 0, -.1], [-s, 0, c, .3],
                         [0, 0, 0, 1.]], device='cuda')
    k = torch.tensor([[32., 0, 32], [0, 32, 32], [0, 0, 1]], device='cuda')
    inputs = dict(means=torch.tensor([[0., 0, 3.]], device='cuda', requires_grad=True),
                  quats=torch.tensor([[1., 0, 0, 0]], device='cuda', requires_grad=True),
                  scales=torch.tensor([[1., 1., 1.]], device='cuda', requires_grad=True),
                  opacities=torch.tensor([.9], device='cuda', requires_grad=True),
                  colors=torch.ones(1, 3, device='cuda'), viewmats=view[None], Ks=k[None],
                  width=64, height=64, packed=False, render_mode='RGB+ED')
    native, alpha, _ = _rasterize('2dgs', **inputs)
    exact, corrected_alpha, _ = _rasterize('2dgs', surface_depth='intersection', **inputs)
    torch.testing.assert_close(native[..., :3], exact[..., :3], rtol=0, atol=0)
    torch.testing.assert_close(alpha, corrected_alpha, rtol=0, atol=0)
    y, x = torch.meshgrid(torch.arange(64, device='cuda')+.5,
                          torch.arange(64, device='cuda')+.5, indexing='ij')
    rays = torch.stack((x, y, torch.ones_like(x)), -1) @ torch.linalg.inv(k).T
    mask = alpha[0, ..., 0] > .1
    points = (rays*exact[0, ..., -1:]-view[:3, 3]) @ view[:3, :3]
    error = (points[..., 2][mask]-3).abs()
    assert error.max() < 1e-5, error.max()
    native_points = (rays*native[0, ..., -1:]-view[:3, 3]) @ view[:3, :3]
    native_error = (native_points[..., 2][mask]-3).abs().mean()
    assert native_error > .1
    # Compare camera-Z derivatives at an off-axis interior pixel to the plane equation.
    from gsplat.utils import normalized_quat_to_rotmat
    n = view[:3, :3] @ normalized_quat_to_rotmat(
        torch.nn.functional.normalize(inputs['quats'], dim=-1))[0, :, 2]
    center = view[:3, :3] @ inputs['means'][0]+view[:3, 3]
    reference = (n @ center)/(rays[32, 40] @ n)
    params = (inputs['means'], inputs['quats'])
    actual_grad = torch.autograd.grad(exact[0, 32, 40, -1], params, retain_graph=True)
    analytic_grad = torch.autograd.grad(reference, params)
    for actual, expected in zip(actual_grad, analytic_grad):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    layered = dict(inputs,
        means=torch.tensor([[0., 0, 3.], [.05, 0, 4.]], device='cuda', requires_grad=True),
        quats=torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]], device='cuda'),
        scales=torch.tensor([[.3, .3, 1.], [.3, .3, 1.]], device='cuda'),
        opacities=torch.tensor([.65, .7], device='cuda', requires_grad=True),
        colors=torch.ones(2, 3, device='cuda'), viewmats=torch.eye(4, device='cuda')[None])
    old, _, _ = _rasterize('2dgs', **layered)
    new, _, _ = _rasterize('2dgs', surface_depth='intersection', **layered)
    # For frontoparallel layers, native depths and their position/opacity gradients
    # are already correct. Check the independent sparse composition against CUDA.
    torch.testing.assert_close(new, old, rtol=2e-5, atol=1e-6)
    params = (layered['means'], layered['opacities'])
    old_grad = torch.autograd.grad(old[0, 30:34, 30:34, -1].mean(), params)
    new_grad = torch.autograd.grad(new[0, 30:34, 30:34, -1].mean(), params)
    for actual, expected in zip(new_grad, old_grad):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    empty = dict(layered, means=torch.tensor([[0., 0, -3.], [0., 0, -4.]], device='cuda'))
    blank, coverage, _ = _rasterize('2dgs', surface_depth='intersection', **empty)
    assert torch.count_nonzero(coverage) == 0 and torch.count_nonzero(blank) == 0
    return {'native_plane_error_mean':float(native_error),
            'intersection_plane_error_max':float(error.max()), 'analytic_geometry_gradients':True,
            'layered_opacity_geometry_gradients':True, 'empty_view':True}


def check_surfel_refinement(g, model, sample):
    count = len(g.params['means'])
    optimizers = g.optimizers()
    strategy = Refinement(refine_start_iter=0, refine_stop_iter=10, refine_every=1,
                          reset_every=100, pause_refine_after_reset=0,
                          grow_grad2d=0., grow_scale3d=0., prune_opa=0., prune_scale3d=float('inf'),
                          max_points=count+8, key_for_gradient='means2d', absgrad=True)
    state = strategy.initialize_state(g.radius)
    image, _, info = render_observation(g, model, sample, 0., True, True, 2.2, 'deep', True)
    strategy.step_pre_backward(g.params, optimizers, state, 1, info)
    image.mean().backward()
    for optimizer in optimizers.values():
        optimizer.step()
    strategy.step_post_backward(g.params, optimizers, state, 1, info)
    assert count < len(g.params['means']) <= count+8
    assert g.params['scales'].shape[1] == 2
    assert all(torch.isfinite(p).all() for p in g.params.values())
    return len(g.params['means'])-count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', default='/workspace/datasets/SSD-GS/data/Real_NRHints/Cat')
    parser.add_argument('--methods', nargs='+', choices=list(METHODS),
                        default=list(METHODS))
    parser.add_argument('--surface-priors', type=Path)
    parser.add_argument('--prepare-priors', action='store_true', help='Run official frozen predictors on the test subset')
    parser.add_argument('--frames', type=int, default=4, help='Evenly spaced real training frames')
    parser.add_argument('--geometry-warmup-steps', type=int, default=0)
    parser.add_argument('--attention-score', choices=['dot', 'cosine'], default='dot')
    parser.add_argument('--material-decoder', type=Path)
    parser.add_argument('--material-checkpoint', type=Path, help='Check neural to analytic GGX transfer on a fitted scene')
    parser.add_argument('--residual-checkpoint', type=Path, help='Check frozen-GS radiance residual training and reload')
    parser.add_argument('--residual-normal', choices=['geometry', 'material'], default='material')
    parser.add_argument('--residual-interaction', choices=['none', 'add', 'multiply'], default='none')
    parser.add_argument('--residual-angular-bank', choices=['none','wide','narrow'], default='none')
    parser.add_argument('--residual-paired-context', action='store_true')
    parser.add_argument('--residual-pair-weight', type=float, default=0.)
    parser.add_argument('--residual-frames', type=int, nargs='+',
                        help='Explicit residual frame pool for each 3-step fit/resume stage')
    parser.add_argument('--optimize-light-scale', action='store_true')
    parser.add_argument('--sdf-checkpoint', type=Path, help='Run the SDF geometry check on a fitted 2DGS checkpoint')
    parser.add_argument('--sdf-volume', action='store_true', help='Check independent SDF image training and reload')
    parser.add_argument('--sdf-volume-only', action='store_true')
    parser.add_argument('--sdf-detail', action='store_true')
    parser.add_argument('--sdf-volume-detail', action='store_true')
    parser.add_argument('--sdf-volume-peak-fraction', type=float, default=0.)
    parser.add_argument('--sdf-volume-rays', type=int, help='Override the SDF smoke training ray budget')
    parser.add_argument('--sdf-volume-samples', type=int, help='Override the SDF smoke coarse and fine sample counts')
    parser.add_argument('--sdf-volume-fixed-sharpness', type=float,
                        help='Set and freeze SDF sharpness during both smoke training phases')
    parser.add_argument('--sdf-volume-peak-context-fraction', type=float, default=0.)
    parser.add_argument('--sdf-volume-hint-encoding', action='store_true')
    parser.add_argument('--sdf-resolution', type=int, help='Resolution of the SDF training smoke check')
    parser.add_argument('--sdf-shading', action='store_true')
    parser.add_argument('--sdf-primitive-weight', type=float, default=0.)
    parser.add_argument('--freeze-sdf', action='store_true')
    parser.add_argument('--normal-field', action='store_true')
    parser.add_argument('--surface-depth', choices=['center', 'intersection'], default='center')
    parser.add_argument('--sdf-start', type=int, default=1)
    parser.add_argument('--optimize-cameras', action='store_true')
    parser.add_argument('--check-budget-extension', action='store_true',
                        help='Check that a longer run preserves its short-run training prefix')
    args = parser.parse_args()
    if args.residual_frames is not None and not args.residual_checkpoint:
        parser.error('--residual-frames requires --residual-checkpoint')
    if args.residual_interaction != 'none' and not args.residual_checkpoint:
        parser.error('--residual-interaction requires --residual-checkpoint')
    if args.residual_angular_bank != 'none' and (not args.residual_checkpoint or
            args.residual_interaction != 'multiply' or args.residual_normal != 'geometry'):
        parser.error('--residual-angular-bank requires --residual-checkpoint with multiply and geometry')
    if args.residual_checkpoint:
        check_radiance_residual_checkpoint(args.residual_checkpoint.resolve(), args.output.resolve(),
                                          args.residual_normal, args.residual_frames, args.residual_interaction,
                                          args.residual_angular_bank, args.residual_paired_context, args.residual_pair_weight)
        return
    if args.sdf_volume_only and not args.sdf_volume:
        parser.error('--sdf-volume-only requires --sdf-volume')
    if args.sdf_volume_samples is not None and (not args.sdf_volume or args.sdf_volume_samples < 4):
        parser.error('--sdf-volume-samples requires --sdf-volume and at least 4 samples')
    if args.sdf_volume_fixed_sharpness is not None and (
        not args.sdf_volume or not math.isfinite(args.sdf_volume_fixed_sharpness) or args.sdf_volume_fixed_sharpness <= 0):
        parser.error('--sdf-volume-fixed-sharpness requires --sdf-volume and a positive finite value')
    if (args.sdf_detail or args.sdf_volume_detail) and not args.sdf_volume:
        parser.error('Detail checks require --sdf-volume and an existing volume checkpoint')
    if args.material_checkpoint:
        check_material_checkpoint(args.material_checkpoint.resolve(), args.output.resolve(), args.optimize_light_scale)
        return
    if args.surface_depth == 'intersection':
        report = check_surfel_depth()
        print(json.dumps(report), flush=True)
    if (args.sdf_shading or args.sdf_primitive_weight or args.freeze_sdf or args.normal_field or args.sdf_volume) and not args.sdf_checkpoint:
        parser.error('SDF options require --sdf-checkpoint')
    if args.sdf_checkpoint:
        check_sdf_checkpoint(args.sdf_checkpoint.resolve(), args.output.resolve(), args.sdf_shading,
                             args.sdf_primitive_weight, args.freeze_sdf, args.normal_field, args.surface_depth,
                             args.sdf_start, args.sdf_volume, args.sdf_volume_only, args.sdf_detail, args.sdf_volume_detail,
                             args.sdf_volume_peak_fraction, args.sdf_resolution,
                             args.sdf_volume_peak_context_fraction, args.sdf_volume_hint_encoding, args.sdf_volume_rays,
                             args.sdf_volume_samples, args.sdf_volume_fixed_sharpness)
        if args.surface_depth == 'intersection':
            (args.output/'depth_report.json').write_text(json.dumps(report, indent=2)+'\n')
        return
    if 'neural_material' in args.methods and args.material_decoder is None:
        parser.error('neural_material requires --material-decoder from pretrain_material.py')
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.surface_priors:
        args.surface_priors = args.surface_priors.resolve()
        args.scene = torch.load(args.surface_priors/'normal.pt', weights_only=False)['scene']
    else:
        original = SceneDataset(args.scene, 'train', 32)
        subset = output / original.scene_path.parent.name / original.scene_path.name
        subset.mkdir(parents=True)
        selected = torch.linspace(0, len(original)-1, args.frames).long().tolist()
        metadata = dict(original.metadata)
        metadata['frames'] = []
        for index in selected:
            frame = dict(original.frames[index])
            frame['file_path'] = str(_frame_path(original.scene_path, frame))
            metadata['frames'].append(frame)
        (subset/'transforms_train.json').write_text(json.dumps(metadata, indent=2)+'\n')
        args.scene = str(subset)
    if args.prepare_priors:
        args.surface_priors = output/'surface_priors'
        for kind, python in [('normal', root/'.venv/bin/python'),
                             ('depth', root/'third_party/da3_env/bin/python')]:
            command = [str(python), str(root/'prepare_surface_priors.py'), '--kind', kind,
                       '--scene', args.scene, '--output', str(args.surface_priors),
                       '--resolution', '32', '--processing-resolution', '512', '--fit-all']
            with (output/f'{kind}.log').open('w') as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    torch.set_num_threads(2)
    report = {}
    if any(METHODS[name].geometry=='2dgs' for name in args.methods):
        check_surfel_channel_padding()
        print('PASS: disk shadow visibility', check_surfel_shadow(), flush=True)
    for name in args.methods:
        run = output / name
        command = [sys.executable, str(root/'train.py'), '--scene', args.scene,
                   '--output', str(run), '--representation', name, '--steps', '3',
                   '--resolution', '32', '--points', '128', '--max-points', '128',
                   '--shadow-start', '1', '--port-start', '1',
                   '--refine-stop', '0', '--validate-every', '0', '--fit-all']
        if args.optimize_cameras:
            command += ['--optimize-cameras', '--camera-start', '1']
        if 'rank' in METHODS[name].cli_fields:
            command += ['--rank', '8']
        if name == 'surface_attention':
            command += ['--attention-score', args.attention_score]
        if name == 'neural_material':
            command += ['--material-decoder', str(args.material_decoder.resolve())]
        if METHODS[name].geometry=='2dgs':
            command += ['--surface-start', '0', '--geometry-warmup-steps', str(args.geometry_warmup_steps)]
            command += ['--surface-depth', args.surface_depth]
            if args.surface_priors:
                command += ['--surface-priors', str(args.surface_priors)]
            else:
                command += ['--normal-weight', '0', '--depth-weight', '0']
        (output/f'{name}.command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (output/f'{name}.train.log').open('w') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        g, model, saved = load_model(run/'last.pt')
        assert saved['step'] == 3 and saved['config']['representation'] == name
        if args.optimize_cameras:
            from cameras import TrainCameraOffsets
            offsets = TrainCameraOffsets(len(saved['fit_indices']), g.radius).cuda()
            offsets.load_state_dict(saved['camera_offsets'])
            assert offsets.raw[1:].abs().sum()>0 and torch.isfinite(offsets.raw).all()
            assert torch.count_nonzero(offsets.raw[0]) == 0
            other = to_device(SceneDataset(args.scene, 'train', 32)[saved['fit_indices'][1]], 'cuda')
            _, coverage, _ = render_observation(g, model, offsets.correct(other, 1),
                                               0., True, True, 2.2, 'deep')
            gradient = torch.autograd.grad((coverage-other['alpha']).square().mean(), offsets.raw)[0]
            assert torch.isfinite(gradient).all() and gradient[1].abs().sum()>0
            del offsets, other, coverage, gradient
        if name == 'neural_material':
            prior = torch.load(args.material_decoder, map_location='cpu', weights_only=False)
            for key, value in model.decoder.state_dict().items():
                torch.testing.assert_close(value.cpu(), prior['decoder'][key], rtol=0, atol=0)
            assert all(not p.requires_grad for p in model.decoder.parameters())
            del prior
        assert [p.name for p in run.glob('*.pt')] == ['last.pt']
        for key, parameter in model.named_parameters():
            assert torch.isfinite(parameter).all(), (name, key)
        sample = to_device(SceneDataset(args.scene, 'train', 32)[0], 'cuda')
        geometry_only = saved['step'] <= saved['config'].get('geometry_warmup_steps', 0)
        if METHODS[name].geometry=='2dgs' and args.surface_priors:
            dataset = SceneDataset(args.scene, 'train', 32)
            targets = SurfacePriors(args.surface_priors, dataset, saved['fit_indices']).frame(0, 'cuda')
            _, alpha, info = render_observation(g, model, sample, 0., True, True, 2.2, 'deep')
            losses = surface_losses(info, alpha, sample, targets, g.radius, .05, .05, .01, .01)
            for term in ['normal_prior', 'depth_prior']:
                grad = torch.autograd.grad(losses[term], g.params['means'], retain_graph=True)[0]
                assert torch.isfinite(grad).all() and grad.abs().sum() > 0, term
            normal_grad = torch.autograd.grad(losses['normal_prior'], g.params['quats'])[0]
            assert torch.isfinite(normal_grad).all() and normal_grad.abs().sum() > 0
        with torch.no_grad():
            first = render_observation(g, model, sample, 0., True, True, 2.2, 'deep', geometry_only=geometry_only)[0]
            assert torch.isfinite(first).all()
            second_g, second_model, _ = load_model(run/'last.pt')
            second = render_observation(second_g, second_model, sample, 0., True, True, 2.2, 'deep', geometry_only=geometry_only)[0]
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        eval_command = [sys.executable, str(root/'evaluate.py'), str(run/'last.pt'),
                        '--split', 'fit', '--limit', '4' if args.optimize_cameras else '1',
                        '--output', str(run/'evaluation')]
        with (output/f'{name}.eval.log').open('w') as log:
            subprocess.run(eval_command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((run/'evaluation/metrics.json').read_text())
        if args.optimize_cameras:
            assert metrics['corrected_fit_frames'] == 4
            print('PASS: fitted camera offsets, fixed gauge, alpha gradients and CLI corrected-fit rendering', flush=True)
        if args.check_budget_extension:
            extended = output/f'{name}_extended'
            extended_command = command.copy()
            extended_command[extended_command.index('--output')+1] = str(extended)
            extended_command[extended_command.index('--steps')+1] = '4'
            extended_command += ['--lr-decay-steps','3','--save-steps','3']
            with (output/f'{name}.extended.log').open('w') as log:
                subprocess.run(extended_command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
            prefix = torch.load(extended/'step_000003.pt',map_location='cuda',weights_only=False)
            for group in ('gaussians','transport'):
                for key,value in saved[group].items():
                    torch.testing.assert_close(prefix[group][key],value,rtol=1e-5,atol=1e-6)
            final = torch.load(extended/'last.pt',map_location='cpu',weights_only=False)
            assert prefix['step']==3 and final['step']==4
            print('PASS: extended budget preserves the 3-step short-run model and saves both requested/final checkpoints',flush=True)
            del prefix,final
        history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
        if METHODS[name].geometry=='2dgs':
            print('PASS: disk densification added', check_surfel_refinement(second_g, second_model, sample), flush=True)
            check_geometry_warmup(second_g, second_model, sample)
            if args.geometry_warmup_steps:
                assert history[0]['stage']=='geometry'
                assert history[-1]['stage']==('geometry' if args.geometry_warmup_steps>=3 else 'relighting')
                assert sum(row.get('event')=='relighting_start' for row in history)==int(args.geometry_warmup_steps<3)
        report[name] = dict(step=saved['step'], final_loss=history[-1]['loss'],
                            reload_pixel_max_abs=float((first-second).abs().max()),
                            metrics=metrics, parameters=sum(p.numel() for p in model.parameters()))
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f'PASS: {name}: CLI train / final checkpoint / reload / CLI evaluation', flush=True)
        del g, model, second_g, second_model, first, second, sample, saved
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

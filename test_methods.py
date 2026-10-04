"""CPU operator regressions for selectable transport methods."""
import argparse
import io
import math
import unittest

import torch
from torch.nn import functional as F

from gaussians import Gaussians
from methods import METHODS, add_method_arguments, build_transport, resolve_config
from methods.base import direction_encoding, quadrature_mass
from surface import depth_normals, surface_losses
from refinement import split_surfels


class MethodTests(unittest.TestCase):
    def test_foundation_explicit_subset_mapping_uses_original_indices(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from PIL import Image
        from foundation_comparison import (foundation_paths, load_foundation_images,
                                           validate_view_mapping)

        frames = [{'file_path': f'./frame_{index:03}.png'} for index in range(5)]
        selected = [1, 4]
        views = [{'frame_index': index, 'name': f'frame_{index:03}',
                  'file_path': frames[index]['file_path']} for index in selected]
        report = {'evaluated_frames': 2, 'views': views}
        mapping = validate_view_mapping(report, frames, frame_indices=selected)
        self.assertEqual(mapping['mode'], 'indices_and_names_verified')
        self.assertEqual(mapping['frames'], 2)
        misleading = {'evaluated_frames': 2,
                      'views': [dict(view, frame_index=ordinal) for ordinal, view in enumerate(views)]}
        with self.assertRaises(ValueError):
            validate_view_mapping(misleading, frames, frame_indices=selected)
        for invalid in ([], [4, 1], [1, 1], [-1, 4], [1, 5]):
            with self.subTest(frame_indices=invalid), self.assertRaises(ValueError):
                validate_view_mapping(report, frames, frame_indices=invalid)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            targets = {index: torch.full((2, 3, 3), 20*index, dtype=torch.uint8) for index in selected}
            predictions = {index: target+3 for index, target in targets.items()}
            for ordinal, original in enumerate(selected):
                pair = torch.cat((targets[original], predictions[original]), 1)
                Image.fromarray(pair.numpy()).save(root/f'pair_{ordinal:03}.png')
            model = {'layout': 'pairs', 'directory': root}
            paths = foundation_paths(model, len(selected))
            self.assertEqual([pair[0].name for pair in paths], ['pair_000.png', 'pair_001.png'])
            for ordinal, original in enumerate(selected):
                prediction, target = load_foundation_images(model, ordinal, (2, 3))
                torch.testing.assert_close(prediction, predictions[original], rtol=0, atol=0)
                torch.testing.assert_close(target, targets[original], rtol=0, atol=0)

    def test_foundation_train_subset_saved_png_comparison(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from PIL import Image
        from foundation_comparison import compare_foundations

        selected = [1, 4]
        scope = 'Synthetic CPU train subset comparison: original frames 1 and 4 only.'
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scene_dir = root/'scene'
            scene_dir.mkdir()
            frames, targets = [], {}
            for index in range(5):
                target = torch.tensor([180+5*index, 45+3*index, 20+2*index],
                                      dtype=torch.uint8).expand(64, 64, 3).clone()
                target[18, 18] = target[42, 42] = 240
                targets[index] = target
                rgba = torch.cat((target, torch.full((64, 64, 1), 255, dtype=torch.uint8)), -1)
                Image.fromarray(rgba.numpy()).save(scene_dir/f'frame_{index:03}.png')
                frames.append({'file_path': f'./frame_{index:03}', 'file_ext': '.png',
                               'transform_matrix': torch.eye(4).tolist(),
                               'pl_pos': [1.+index, .5, 2.], 'pl_intensity': [1., 1., 1.]})
            (scene_dir/'transforms_train.json').write_text(json.dumps(
                {'camera_intrinsics': [32., 32., 55., 55.], 'frames': frames}))
            models = {}
            for label, changes in (('control', [4, 40]), ('candidate', [2, 20])):
                model_dir = root/label
                model_dir.mkdir()
                for ordinal, (original, change) in enumerate(zip(selected, changes)):
                    target = targets[original]
                    prediction = target.clone()
                    background = target[..., 0] != 240
                    prediction[background] = (prediction[background].short()
                                              + torch.tensor([0, change, change])).byte()
                    Image.fromarray(torch.cat((target, prediction), 1).numpy()).save(
                        model_dir/f'pair_{ordinal:03}.png')
                metric_path = model_dir/'metrics.json'
                metric_path.write_text(json.dumps({'split': 'fit', 'evaluated_frames': 2,
                    'views': [{'frame_index': index, 'name': f'frame_{index:03}',
                               'file_path': frames[index]['file_path']} for index in selected]}))
                models[label] = {'layout': 'pairs', 'directory': str(model_dir),
                                 'metrics': str(metric_path), 'primary_eligible': True,
                                 'provenance': 'Analytic temporary PNG fixture'}
            scene = {'name': 'subset', 'dataset': str(scene_dir), 'split': 'train', 'resolution': 64,
                     'background': 0., 'display_gamma': 2.2, 'frame_indices': selected,
                     'crop_frames': [4], 'control': 'control', 'models': models}
            manifest = {'scope': scope, 'scenes': [scene]}
            manifest_path = root/'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            missing_indices = dict(scene)
            del missing_indices['frame_indices']
            invalid_path = root/'missing_indices.json'
            invalid_path.write_text(json.dumps({'scenes': [missing_indices]}))
            invalid_output = root/'invalid_output'
            invalid_output.mkdir()
            with self.assertRaisesRegex(ValueError, 'frame_indices'):
                compare_foundations(invalid_path, invalid_output, device='cpu', perceptual=False)

            output = root/'comparison'
            output.mkdir()
            summary = compare_foundations(manifest_path, output, device='cpu', perceptual=False)
            scene_output = output/'subset'
            metrics = json.loads((scene_output/'metrics.json').read_text())
            mapping = json.loads((scene_output/'metadata_mapping.json').read_text())
            target_checks = json.loads((scene_output/'target_checks.json').read_text())
            crops = json.loads((scene_output/'crops.json').read_text())
            gates = json.loads((scene_output/'gates.json').read_text())
            self.assertEqual(summary['scope'], scope)
            self.assertEqual(summary['scenes']['subset']['evaluated_frames'], 2)
            self.assertEqual(metrics['evaluated_frames'], 2)
            self.assertEqual([row['frame_index'] for row in mapping['frames']], selected)
            self.assertEqual([row['png_ordinal'] for row in mapping['frames']], [0, 1])
            for label in models:
                self.assertEqual([row['frame_index'] for row in metrics['views'][label]], selected)
                self.assertEqual([row['frame_index'] for row in target_checks['views'][label]], selected)
                self.assertTrue(target_checks['models'][label]['all_saved_targets_exact'])
                self.assertEqual(target_checks['models'][label]['differing_channel_count'], 0)
                self.assertEqual(target_checks['models'][label]['pixel_count'], 2*64*64)
                self.assertEqual([Path(row['models'][label]['prediction']).name for row in mapping['frames']],
                                 ['pair_000.png', 'pair_001.png'])
                rows = metrics['views'][label]
                fixed = metrics['models'][label]['fixed_frame_ring']
                self.assertEqual(fixed, rows[1]['ring'])
                self.assertEqual(fixed['pixels'], 240)
                self.assertEqual(metrics['models'][label]['full_frame_ring']['pixels'], 480)
                self.assertGreater(fixed['neutral_positive_error'], 5*rows[0]['ring']['neutral_positive_error'])
            self.assertEqual([row['frame_index'] for row in crops['selection']], [4])
            self.assertEqual(len(crops['crops']), 2)
            self.assertEqual({row['frame_index'] for row in crops['crops']}, {4})
            self.assertEqual(gates['models']['candidate']['scope'], scope)
            self.assertEqual(summary['scenes']['subset']['gates']['candidate']['scope'], scope)

    def test_foundation_png_orientation_and_exact_names(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from PIL import Image
        from foundation_comparison import foundation_paths, load_foundation_images

        target = torch.tensor([[[0, 255, 17], [2, 3, 4]],
                               [[5, 6, 7], [8, 9, 10]]], dtype=torch.uint8)
        prediction = torch.tensor([[[255, 0, 19], [20, 21, 22]],
                                   [[23, 24, 25], [26, 27, 28]]], dtype=torch.uint8)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root/'pairs'
            pairs.mkdir()
            pair_path = pairs/'pair_000.png'
            Image.fromarray(torch.cat((target, prediction), 1).numpy()).save(pair_path)
            pair_model = {'layout': 'pairs', 'directory': pairs}
            self.assertEqual(foundation_paths(pair_model, 1), [(pair_path, pair_path)])
            pair_prediction, pair_target = load_foundation_images(pair_model, 0, (2, 2))
            for actual, expected in ((pair_prediction, prediction), (pair_target, target)):
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertEqual(actual.device.type, 'cpu')
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'native size'):
                load_foundation_images(pair_model, 0, (2, 3))
            extra_pair = pairs/'pair_001.png'
            Image.fromarray(torch.cat((target, prediction), 1).numpy()).save(extra_pair)
            with self.assertRaisesRegex(ValueError, 'extra=.*pair_001'):
                foundation_paths(pair_model, 1)
            pair_path.unlink()
            with self.assertRaisesRegex(ValueError, 'missing=.*pair_000'):
                foundation_paths(pair_model, 1)

            prediction_dir, target_dir = root/'prediction', root/'target'
            prediction_dir.mkdir()
            target_dir.mkdir()
            prediction_path, target_path = prediction_dir/'00000.png', target_dir/'00000.png'
            Image.fromarray(prediction.numpy()).save(prediction_path)
            Image.fromarray(target.numpy()).save(target_path)
            separate_model = {'layout': 'separate', 'prediction_dir': prediction_dir,
                              'target_dir': target_dir}
            self.assertEqual(foundation_paths(separate_model, 1), [(prediction_path, target_path)])
            separate_prediction, separate_target = load_foundation_images(separate_model, 0, (2, 2))
            torch.testing.assert_close(separate_prediction, prediction, rtol=0, atol=0)
            torch.testing.assert_close(separate_target, target, rtol=0, atol=0)
            extra_prediction = prediction_dir/'00001.png'
            Image.fromarray(prediction.numpy()).save(extra_prediction)
            with self.assertRaisesRegex(ValueError, 'extra=.*00001'):
                foundation_paths(separate_model, 1)
            extra_prediction.unlink()
            target_path.rename(target_dir/'00001.png')
            with self.assertRaisesRegex(ValueError, 'missing=.*00000'):
                foundation_paths(separate_model, 1)

    def test_foundation_frame_mapping_and_unsigned_target_difference(self):
        from foundation_comparison import target_difference, validate_view_mapping

        frames = [{'file_path': './test/image_017.png'}, {'file_path': './test/image_003.png'}]
        views = [{'frame_index': 0, 'name': 'image_017'}, {'frame_index': 1, 'name': 'image_003'}]
        report = {'evaluated_frames': 2, 'views': views}
        self.assertEqual(validate_view_mapping(report, frames),
                         {'mode': 'indices_and_names_verified', 'frames': 2})
        self.assertEqual(validate_view_mapping({'frames': 2}, frames)['mode'], 'count_only')
        invalid_reports = [
            {'evaluated_frames': 1, 'views': views},
            {'evaluated_frames': 2, 'views': views[:1]},
            {'evaluated_frames': 2, 'views': [views[1], views[0]]},
            {'evaluated_frames': 2, 'views': [dict(views[0], name='image_003'), views[1]]},
            {'evaluated_frames': 2, 'views': [dict(views[0], frame_index=1), views[1]]},
            {'evaluated_frames': 2, 'views': [dict(views[0], file_path='./other/image_017.png'), views[1]]},
        ]
        for invalid_report in invalid_reports:
            with self.subTest(report=invalid_report), self.assertRaises(ValueError):
                validate_view_mapping(invalid_report, frames)

        canonical = torch.tensor([[[0, 255, 1], [100, 0, 255]]], dtype=torch.uint8)
        saved = torch.tensor([[[255, 0, 1], [102, 0, 255]]], dtype=torch.uint8)
        difference = target_difference(saved, canonical)
        self.assertEqual(difference['max_abs_byte_difference'], 255)
        self.assertEqual(difference['absolute_byte_difference_sum'], 512)
        self.assertEqual(difference['differing_channel_count'], 3)
        self.assertEqual(difference['differing_pixel_count'], 2)
        self.assertEqual(difference['channel_count'], 6)
        self.assertEqual(difference['pixel_count'], 2)
        self.assertAlmostEqual(difference['mean_abs_byte_difference'], 512/6)
        identity = target_difference(canonical, canonical)
        self.assertEqual(identity['differing_channel_count'], 0)
        self.assertEqual(identity['absolute_byte_difference_sum'], 0)

    def test_foundation_ring_domain_and_pixel_pooling(self):
        from foundation_comparison import aggregate_ring, ring_error

        target = torch.tensor([.75, .25, .125]).expand(17, 25, 3).clone()
        target[8, 7] = .875
        alpha = torch.ones(17, 25, 1)
        alpha[8, 2] = .9
        alpha[8, 12] = 0.
        identity = ring_error(target, target, alpha)
        self.assertEqual(identity['pixels'], 118)
        self.assertEqual(identity['neutral_positive_error_sum'], 0.)
        self.assertEqual(identity['neutral_positive_error'], 0.)

        prediction = torch.tensor([.75, .375, .375]).expand_as(target).clone()
        prediction[8, 7] = 1.  # The GT peak itself is excluded from the ring.
        prediction[8, 20] = 1.  # A false bright spot outside the GT ring is excluded.
        first = ring_error(prediction, target, alpha)
        self.assertEqual(first['pixels'], 118)
        self.assertEqual(first['neutral_positive_error_sum'], 29.5)
        self.assertEqual(first['neutral_positive_error'], .25)

        target[8, 9] = .875
        prediction = torch.tensor([.75, .625, .625]).expand_as(target).clone()
        second = ring_error(prediction, target, torch.ones_like(alpha))
        self.assertEqual(second['pixels'], 141)
        self.assertEqual(second['neutral_positive_error_sum'], 70.5)
        self.assertEqual(second['neutral_positive_error'], .5)
        pooled = aggregate_ring([first, second])
        self.assertEqual(pooled['pixels'], 259)
        self.assertEqual(pooled['neutral_positive_error_sum'], 100.)
        self.assertAlmostEqual(pooled['neutral_positive_error'], 100/259)

        empty_target = torch.tensor([.75, .25, .125]).expand_as(target)
        empty = ring_error(prediction, empty_target, torch.ones_like(alpha))
        self.assertEqual(empty['pixels'], 0)
        self.assertEqual(empty['neutral_positive_error_sum'], 0.)
        self.assertIsNone(empty['neutral_positive_error'])
        self.assertEqual(aggregate_ring([first, empty, second]), pooled)
        self.assertEqual(aggregate_ring([empty]), empty)

    def test_residual_pair_local_pools_and_budget(self):
        from residual_sampling import build_residual_pair_pool, select_residual_pairs
        height, width = 17, 19
        alpha = torch.ones(height*width)
        alpha[[1, 19, 20]] = torch.tensor([.9, .89, .91])
        peaks = torch.zeros(height*width, dtype=torch.bool)
        peaks[[0, 4*width+4, 8*width+8, 16*width+18]] = True
        valid = torch.arange(height*width)
        valid = valid[(valid != 2) & (valid != 8*width+8)]
        rays = {'valid': valid, 'target': alpha, 'shape': (height, width)}
        pool = build_residual_pair_pool(rays, peaks.reshape(height, width))
        self.assertEqual(pool['stats']['gt_peak_pixels'], 4)
        self.assertEqual(pool['stats']['valid_peak_pixels'], 3)
        self.assertEqual(pool['stats']['eligible_anchors'], 3)
        valid_set = set(valid.tolist())
        for row, anchor in enumerate(pool['anchors'].tolist()):
            ay, ax = divmod(anchor, width)
            expected = {y*width+x for y in range(max(0, ay-5), min(height, ay+6))
                        for x in range(max(0, ax-5), min(width, ax+6))
                        if y*width+x in valid_set and not peaks[y*width+x]
                        and alpha[y*width+x] > .9}
            begin, count = int(pool['ring_offsets'][row]), int(pool['ring_counts'][row])
            self.assertEqual(set(pool['rings'][begin:begin+count].tolist()), expected)
        for count in (2048, 20):
            indices, pairs = select_residual_pairs(
                rays, count, torch.Generator().manual_seed(7), pool)
            slots = count//4
            self.assertEqual(indices.shape, (count,))
            self.assertEqual(pairs.shape, (slots, 2))
            torch.testing.assert_close(indices[:slots], pairs[:, 0])
            torch.testing.assert_close(indices[slots:2*slots], pairs[:, 1])
            self.assertTrue(peaks[pairs[:, 0]].all())
            self.assertFalse(peaks[pairs[:, 1]].any())
            self.assertTrue((alpha[pairs[:, 1]] > .9).all())
            self.assertTrue((alpha[indices[2*slots:3*slots]] > .9).all())
            self.assertTrue(set(indices.tolist()).issubset(valid_set))
            self.assertTrue(((pairs[:, 0]//width-pairs[:, 1]//width).abs() <= 5).all())
            self.assertTrue(((pairs[:, 0]%width-pairs[:, 1]%width).abs() <= 5).all())
        for count in (0, 3, 2050):
            with self.assertRaisesRegex(ValueError, 'divisible by four'):
                select_residual_pairs(rays, count, torch.Generator(), pool)

    def test_residual_pair_anchor_pixel_distribution_and_component_audit(self):
        from residual_sampling import (build_residual_pair_pool, select_residual_pairs,
                                       residual_pair_sample_stats)
        height, width = 17, 40
        peaks = torch.zeros(height, width, dtype=torch.bool)
        peaks[2, 2] = True
        peaks[2:4, 14:16] = True
        peaks[9, 2:7] = True
        peaks[10:14, 25:30] = True
        rays = {'valid': torch.arange(height*width), 'target': torch.ones(height*width),
                'shape': (height, width)}
        pool = build_residual_pair_pool(rays, peaks)
        self.assertEqual(pool['stats']['component_buckets']['1_to_4'],
                         {'components': 2, 'gt_peak_pixels': 5, 'eligible_components': 2,
                          'eligible_anchors': 5})
        self.assertEqual(pool['stats']['component_buckets']['5_to_16']['gt_peak_pixels'], 5)
        self.assertEqual(pool['stats']['component_buckets']['gt16']['gt_peak_pixels'], 20)
        first = select_residual_pairs(rays, 32768, torch.Generator().manual_seed(3), pool)
        second = select_residual_pairs(rays, 32768, torch.Generator().manual_seed(3), pool)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        pairs = first[1]
        small_count = int((pairs[:, 0] == 2*width+2).sum())
        four_count = int(((pairs[:, 0]//width >= 2) & (pairs[:, 0]//width < 4)
                          & (pairs[:, 0]%width >= 14) & (pairs[:, 0]%width < 16)).sum())
        self.assertGreater(four_count/small_count, 3.3)
        self.assertLess(four_count/small_count, 4.7)
        audit = residual_pair_sample_stats(pool, pairs)['component_buckets']
        self.assertEqual(sum(row['anchor_draws'] for row in audit.values()), 8192)
        self.assertEqual([row['sampled_peak_pixels'] for row in audit.values()], [5, 5, 20])
        self.assertEqual([len(row['component_ids']) for row in audit.values()], [2, 1, 1])

    def test_residual_pair_excludes_anchors_without_their_own_ring(self):
        from residual_sampling import build_residual_pair_pool, select_residual_pairs
        height, width = 7, 20
        peaks = torch.zeros(height, width, dtype=torch.bool)
        peaks[3, 3] = peaks[3, 16] = True
        alpha = peaks.flatten().float()
        alpha[3*width+17] = 1.
        rays = {'valid': torch.arange(height*width), 'target': alpha, 'shape': (height, width)}
        pool = build_residual_pair_pool(rays, peaks)
        self.assertEqual(pool['stats']['valid_peak_pixels'], 2)
        self.assertEqual(pool['stats']['eligible_anchors'], 1)
        self.assertEqual(pool['stats']['component_buckets']['1_to_4']['eligible_components'], 1)
        torch.testing.assert_close(pool['anchors'], torch.tensor([3*width+16]))
        indices, pairs = select_residual_pairs(rays, 2048, torch.Generator().manual_seed(3), pool)
        torch.testing.assert_close(pairs[:, 0], torch.full((512,), 3*width+16))
        torch.testing.assert_close(pairs[:, 1], torch.full((512,), 3*width+17))
        self.assertEqual(len(indices), 2048)

    def test_residual_pair_degenerate_frames_preserve_existing_fallback(self):
        from residual_sampling import (build_residual_pair_pool, select_residual_pairs,
                                       residual_pair_sample_stats)
        from sdf_volume import select_rays
        height, width = 6, 7
        valid = torch.arange(1, height*width-1)
        for peaks, alpha in ((torch.zeros(height, width, dtype=torch.bool), torch.ones(height*width)),
                             (torch.ones(height, width, dtype=torch.bool), torch.ones(height*width)),
                             (torch.zeros(height, width, dtype=torch.bool), torch.zeros(height*width))):
            rays = {'valid': valid, 'target': alpha, 'shape': (height, width)}
            pool = build_residual_pair_pool(rays, peaks)
            indices, pairs = select_residual_pairs(rays, 2048, torch.Generator().manual_seed(8), pool)
            baseline = select_rays(rays, 2048, torch.Generator().manual_seed(8))
            torch.testing.assert_close(indices, baseline, rtol=0, atol=0)
            self.assertEqual(pairs.shape, (0, 2))
            self.assertEqual(pool['stats']['eligible_anchors'], 0)
            self.assertEqual(sum(row['anchor_draws'] for row in
                                 residual_pair_sample_stats(pool, pairs)['component_buckets'].values()), 0)
            self.assertLess(indices.unique().numel(), len(indices))

    def test_residual_pair_repeats_raw_rgb_and_both_endpoint_gradients(self):
        from residual_sampling import residual_pair_loss
        predicted = torch.tensor([[2., -3., 4.], [0., 0., 0.], [5., 6., -7.]], requires_grad=True)
        target = torch.zeros_like(predicted)
        pairs = torch.tensor([[0, 1], [0, 1], [2, 1]])
        loss, supported = residual_pair_loss(predicted, target, torch.ones(3), pairs, 4)
        self.assertTrue(supported.all())
        torch.testing.assert_close(loss, torch.tensor((2*9+18)/12))
        loss.backward()
        expected = torch.tensor([[2., -2., 2.], [-3., 1., -1.], [1., 1., -1.]])/12
        torch.testing.assert_close(predicted.grad, expected)
        draws = torch.tensor([0, 2, 2, 1])
        rgb = (predicted[draws]-target[draws]).abs().mean()
        self.assertNotEqual(float(rgb), float((predicted-target).abs().mean()))
        torch.testing.assert_close(rgb, torch.tensor((9+2*18)/12))

    def test_residual_pair_support_uses_both_ends_and_fixed_denominator(self):
        from residual_sampling import residual_pair_loss
        predicted = torch.tensor([[3., 2., 1.], [0., 0., 0.], [5., 5., 5.], [9., 9., 9.]],
                                 requires_grad=True)
        target = torch.zeros_like(predicted)
        pairs = torch.tensor([[0, 1], [0, 2], [3, 1], [0, 1]])
        loss, supported = residual_pair_loss(predicted, target, torch.tensor([1., 1., 0., -1.]), pairs, 4)
        torch.testing.assert_close(supported, torch.tensor([True, False, False, True]))
        torch.testing.assert_close(loss, torch.tensor(1.))
        loss.backward()
        torch.testing.assert_close(predicted.grad,
                                   torch.tensor([[1., 1., 1.], [-1., -1., -1.], [0., 0., 0.], [0., 0., 0.]])/6)
        for tested_pairs, alpha in ((pairs, torch.zeros(4)), (pairs[:0], torch.ones(4))):
            predicted.grad = None
            zero, supported = residual_pair_loss(predicted, target, alpha, tested_pairs, 4)
            self.assertTrue(zero.requires_grad)
            self.assertEqual(float(zero), 0.)
            self.assertEqual(int(supported.sum()), 0)
            zero.backward()
            torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))

    def test_residual_pair_lambda_zero_matches_rgb_control(self):
        from residual_sampling import residual_pair_loss
        target = torch.tensor([[.91, .74, .67], [.23, .17, .19]])
        first = torch.tensor([[.7, .8, .5], [.35, .25, .3]], requires_grad=True)
        second = first.detach().clone().requires_grad_()
        draws = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1])
        pairs = torch.tensor([[0, 1], [0, 1]])
        rgb_first = (first[draws]-target[draws]).abs().mean()
        rgb_second = (second[draws]-target[draws]).abs().mean()
        pair_first, _ = residual_pair_loss(first, target, torch.ones(2), pairs, 2)
        pair_second, _ = residual_pair_loss(second, target, torch.ones(2), pairs, 2)
        zero_control = rgb_first+0*pair_first
        torch.testing.assert_close(zero_control, rgb_second, rtol=0, atol=0)
        torch.testing.assert_close(pair_first, pair_second, rtol=0, atol=0)
        control_grad = torch.autograd.grad(zero_control, first, retain_graph=True)[0]
        torch.testing.assert_close(control_grad, torch.autograd.grad(rgb_second, second, retain_graph=True)[0],
                                   rtol=0, atol=0)
        self.assertGreater(float(rgb_second+.25*pair_second), float(zero_control))

    def test_sdf_peak_ray_quota_and_empty_frame(self):
        from sdf_volume import select_rays
        rays = {'valid':torch.arange(2, 100), 'target':torch.cat((torch.zeros(20), torch.ones(80)))}
        peaks = torch.zeros(100, dtype=torch.bool)
        peaks[[0, 30, 70]] = True  # Domain-invalid pixel 0 must never be sampled.
        gen = lambda: torch.Generator().manual_seed(7)
        selected = select_rays(rays, 512, gen(), peaks, .25)
        self.assertEqual(len(selected), 512)
        self.assertTrue(peaks[selected[:128]].all())
        self.assertTrue((selected >= 2).all())
        self.assertTrue((rays['target'][selected[128:320]] > .9).all())
        baseline = select_rays(rays, 512, gen())
        torch.testing.assert_close(select_rays(rays, 512, gen(), peaks, 0.), baseline, rtol=0, atol=0)
        torch.testing.assert_close(select_rays(rays, 512, gen(), torch.zeros_like(peaks), .25), baseline, rtol=0, atol=0)
        context = torch.zeros_like(peaks)
        context[[29, 31, 69, 71]] = True
        selected = select_rays(rays, 512, gen(), peaks, .25, context, .25)
        self.assertEqual(len(selected), 512)
        self.assertTrue(peaks[selected[:128]].all())
        self.assertTrue(context[selected[128:256]].all())
        self.assertTrue((rays['target'][selected[256:384]] > .9).all())

    def test_sdf_hint_encoding_zero_expansion_and_gradients(self):
        from sdf_volume import SDFRadiance, encode_hints
        torch.manual_seed(3)
        old, expanded = SDFRadiance(), SDFRadiance(hint_encoding=True)
        expanded.load_state_dict({**expanded.state_dict(), **old.state_dict()})
        points = torch.randn(12, 3)
        normals = F.normalize(torch.randn(12, 3), dim=-1)
        wi = F.normalize(torch.randn(12, 3), dim=-1)
        wo = F.normalize(torch.randn(12, 3), dim=-1)
        light = torch.tensor([.2, .7, -1.])
        before, after = old(points, normals, wi, wo, light), expanded(points, normals, wi, wo, light)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        after.sum().backward()
        self.assertGreater(float(expanded.hint_encoding.weight.grad.abs().sum()), 0.)
        self.assertEqual(expanded.hint_encoding.weight.numel(), 3072)
        zero = encode_hints(torch.zeros(1, 3))
        torch.testing.assert_close(zero, torch.cat((torch.zeros(1, 12), torch.ones(1, 12)), -1), rtol=0, atol=0)

    def test_zero_detail_expansion_preserves_field_normals_and_radiance(self):
        from sdf import SurfaceSDF
        from sdf_volume import SDFRadiance
        coarse = SurfaceSDF(torch.zeros(3), 1.)
        detailed = SurfaceSDF(torch.zeros(3), 1., detail=True)
        with torch.no_grad():
            coarse.network[-1].bias.fill_(.07)
        detailed.load_state_dict({**detailed.state_dict(), **coarse.state_dict()})
        points = torch.randn(8, 3).clamp(-.8, .8).requires_grad_()
        before, after = coarse(points), detailed(points)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        n0 = torch.autograd.grad(before.sum(), points)[0]
        n1 = torch.autograd.grad(after.sum(), points)[0]
        torch.testing.assert_close(n0, n1, rtol=0, atol=0)
        head0, head1 = SDFRadiance(), SDFRadiance(detail=True)
        with torch.no_grad():
            head0.network[-1].weight.normal_(std=.08)
            head0.network[-1].bias.copy_(torch.tensor([.1, -.2, .3]))
        head1.load_state_dict({**head1.state_dict(), **head0.state_dict()})
        wi, wo = F.normalize(torch.randn(8, 3), dim=-1), F.normalize(torch.randn(8, 3), dim=-1)
        n = F.normalize(n0, dim=-1); light = torch.tensor([.3, 1., -2.])
        torch.testing.assert_close(head0(points, n, wi, wo, light), head1(points, n, wi, wo, light), rtol=0, atol=0)

    def test_spatial_grid_interpolation_and_normal_loss_gradients(self):
        from materials.spatial_detail import MultiResolutionGrid, SpatialDetail
        grid = MultiResolutionGrid(levels=1, base_resolution=4, max_resolution=4, log2_table_size=8).double()
        vertices = torch.cartesian_prod(*[torch.linspace(-1, 1, 4, dtype=torch.double) for _ in range(3)])
        with torch.no_grad():
            grid.table.copy_(torch.stack((vertices[:, 0], vertices[:, 1]+vertices[:, 2]), -1))
        points = torch.tensor([[-.317, .233, .127], [.411, -.123, .377]], dtype=torch.double, requires_grad=True)
        expected = torch.stack((points[:, 0], points[:, 1]+points[:, 2]), -1)
        torch.testing.assert_close(grid(points), expected, rtol=0, atol=1e-12)
        detail = SpatialDetail(1, levels=3, base_resolution=4, max_resolution=16, log2_table_size=8).double()
        torch.testing.assert_close(detail(points), torch.zeros(2, 1, dtype=torch.double), rtol=0, atol=0)
        with torch.no_grad():
            detail.projection.weight.fill_(.3)
        self.assertTrue(torch.autograd.gradcheck(lambda x: detail(x), (points,)))
        self.assertTrue(torch.autograd.gradgradcheck(lambda x: detail(x), (points,)))
        normal_gradient = torch.autograd.grad(detail(points).sum(), points, create_graph=True)[0]
        normal_gradient.square().sum().backward()
        self.assertTrue(torch.isfinite(detail.grid.table.grad).all())
        self.assertGreater(float(detail.grid.table.grad.abs().sum()), 0.)

    def test_spatial_sdf_residual_vanishes_at_domain_boundary(self):
        from materials.spatial_detail import SpatialDetail
        detail = SpatialDetail(1, boundary_zero=True, levels=2, base_resolution=4, max_resolution=8, log2_table_size=8)
        with torch.no_grad():
            detail.projection.weight.fill_(1.)
        points = torch.tensor([[1., .2, .3], [0., -1., .4], [.1, .2, 1.], [2., 0., 0.]], requires_grad=True)
        result = detail(points)
        gradient = torch.autograd.grad(result.sum(), points, create_graph=True)[0]
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)

    def test_sdf_importance_sampling_explores_empty_coarse_intervals(self):
        from sdf import sdf_ray_weights
        from sdf_volume import interval_edges
        # A real analytic slab lies between coarse samples. Uniform exploration
        # must introduce new locations, rather than duplicate old boundaries.
        class Slab(torch.nn.Module):
            def forward(self, points):
                return (points[..., 2]-.25).abs()-.01
        field = Slab()
        rays = {'origin':torch.tensor([0., 0., -2.]), 'direction':torch.tensor([[0., 0., 1.]]),
                'near':torch.tensor([1.]), 'far':torch.tensor([3.])}
        edges = interval_edges(field, rays, torch.tensor([0]), 4, 1e4)
        points = rays['origin']+rays['direction'][:, None]*edges[..., None]
        self.assertLess(float(field(points).min()), 0.)
        _, opacity = sdf_ray_weights(field(points), 1e4)
        torch.testing.assert_close(opacity, torch.ones(1), rtol=0, atol=1e-6)

    def test_sdf_volume_light_linearity_and_field_detachment(self):
        from sdf import SurfaceSDF, silhouette_rays
        from sdf_volume import SDFRadiance, render_rays
        field = SurfaceSDF(torch.zeros(3), 1.)
        radiance = SDFRadiance()
        view = torch.eye(4); view[2, 3] = 2
        c2w = torch.eye(4); c2w[2, 3] = -2
        sample = {'alpha':torch.ones(16, 16, 1), 'c2w':c2w, 'viewmat':view,
                  'K':torch.tensor([[8., 0, 8], [0, 8., 8], [0, 0, 1.]]),
                  'light_pos':torch.tensor([-.4, .8, -1.]), 'light_intensity':torch.ones(3)}
        rays = silhouette_rays(sample, field)
        ids = torch.tensor([119, 120, 135, 136])
        with torch.no_grad():
            reference = render_rays(field, radiance, sample, rays, ids, 1., samples=16)
            bright = render_rays(field, radiance, dict(sample, light_intensity=3*sample['light_intensity']), rays, ids, 1., samples=16)
        torch.testing.assert_close(bright['linear'], 3*reference['linear'], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(bright['depth'], reference['depth'], rtol=0, atol=0)
        for detached in [True, False]:
            field.zero_grad(set_to_none=True); radiance.zero_grad(set_to_none=True)
            result = render_rays(field, radiance, sample, rays, ids, 1., samples=16, detach_field=detached)
            result['linear'].square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in radiance.parameters() if p.grad is not None))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in radiance.parameters() if p.grad is not None), 0)
            if detached:
                self.assertTrue(all(p.grad is None for p in field.parameters()))
            else:
                gradients = [p.grad for p in field.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_sdf_volume_mutual_gradient_targets(self):
        from sdf import SurfaceSDF, silhouette_rays
        from sdf_volume import SDFRadiance, render_rays, ray_geometry_losses
        field = SurfaceSDF(torch.zeros(3), 1.); radiance = SDFRadiance()
        view = torch.eye(4); view[2, 3] = 2
        c2w = torch.eye(4); c2w[2, 3] = -2
        sample = {'alpha':torch.ones(16, 16, 1), 'c2w':c2w, 'viewmat':view,
                  'K':torch.tensor([[8., 0, 8], [0, 8., 8], [0, 0, 1.]]),
                  'light_pos':torch.tensor([-.4, .8, -1.]), 'light_intensity':torch.ones(3)}
        rays = silhouette_rays(sample, field); ids = torch.tensor([119, 120, 135, 136])
        depth = torch.nn.Parameter(torch.full((16, 16, 1), 1.6))
        normal = torch.nn.Parameter(torch.tensor([.1, 0., -1.]).expand(16, 16, 3).clone())
        info = {'surface_depth':depth, 'surface_normals':normal}
        for destination in ['field', 'gs']:
            field.zero_grad(set_to_none=True); radiance.zero_grad(set_to_none=True)
            depth.grad = normal.grad = None
            result = render_rays(field, radiance, sample, rays, ids, 1., samples=16)
            losses = ray_geometry_losses(result, ids, info, torch.ones(16, 16, 1), sample, 1., .05, .01)
            (losses['sdf_ray_depth_to_'+destination]+losses['sdf_ray_normal_to_'+destination]).backward()
            if destination == 'field':
                self.assertIsNone(depth.grad); self.assertIsNone(normal.grad)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in field.parameters() if p.grad is not None), 0)
            else:
                self.assertTrue(all(p.grad is None for p in field.parameters()))
                self.assertTrue(all(p.grad is None for p in radiance.parameters()))
                self.assertGreater(float(depth.grad.abs().sum()), 0)
                self.assertGreater(float(normal.grad.abs().sum()), 0)

    def test_sdf_cdf_plane_depth_and_surface_gradient(self):
        from sdf import sdf_ray_weights
        distance = torch.linspace(-1, 1, 1025, dtype=torch.double)
        plane = torch.tensor(.13, dtype=torch.double, requires_grad=True)
        weights, opacity = sdf_ray_weights((plane-distance)[None], 128.)
        midpoint = (distance[1:]+distance[:-1])*.5
        depth = (weights*midpoint).sum()/opacity[0]
        torch.testing.assert_close(depth, plane, rtol=0, atol=2e-5)
        torch.testing.assert_close(torch.autograd.grad(depth, plane)[0], torch.ones_like(plane), rtol=0, atol=2e-4)
        torch.testing.assert_close(opacity, torch.ones_like(opacity), rtol=0, atol=1e-12)
        reverse, reverse_opacity = sdf_ray_weights((distance-plane)[None], 128.)
        self.assertEqual(float(reverse.sum()), 0.)
        self.assertEqual(float(reverse_opacity), 0.)

    def test_sdf_cdf_occlusion_empty_rays_and_extreme_values(self):
        from sdf import sdf_ray_weights
        edges = torch.linspace(-1, 1, 2049, dtype=torch.double)
        sdf = torch.minimum((edges+.5).abs()-.1, (edges-.45).abs()-.15)
        weights, opacity = sdf_ray_weights(sdf[None], 512.)
        mid = .5*(edges[:-1]+edges[1:])
        depth = (weights*mid).sum()/opacity[0]
        torch.testing.assert_close(depth, edges.new_tensor(-.6), rtol=0, atol=2e-5)
        self.assertLess(float(weights[..., mid>0].sum()), 1e-6)
        empty, empty_opacity = sdf_ray_weights(torch.ones(2, 64), 1e4)
        self.assertEqual(float(empty.sum()), 0.)
        self.assertEqual(float(empty_opacity.sum()), 0.)
        extreme = torch.tensor([[10., -10., -10., 10.]], requires_grad=True)
        weights, opacity = sdf_ray_weights(extreme, 1e4)
        (weights.sum()+opacity.sum()).backward()
        self.assertTrue(torch.isfinite(weights).all() and torch.isfinite(extreme.grad).all())
        torch.testing.assert_close(weights, torch.tensor([[1., 0., 0.]]), rtol=0, atol=0)

    def test_analytic_material_peak_reciprocity_and_gradients(self):
        import math
        from materials.ggx import GGXMaterial
        material = GGXMaterial()
        code = torch.tensor([[.04, .3, .8, 0., 0., .7]], dtype=torch.double, requires_grad=True)
        normal = torch.tensor([[0., 0., 1.]], dtype=torch.double)
        peak, _, _ = material(code, normal, normal)
        torch.testing.assert_close(peak, code[:, :3] / (4 * math.pi * .001**2))
        wi = F.normalize(torch.tensor([[.4, .1, 1.]], dtype=torch.double), dim=-1)
        wo = F.normalize(torch.tensor([[-.3, .2, 1.]], dtype=torch.double), dim=-1)
        sample = torch.tensor([[.04, .3, .8, .45, .8, .7]], dtype=torch.double, requires_grad=True)
        response, transmission, _ = material(sample, wi, wo, normal)
        reverse, _, _ = material(sample, wo, wi, normal)
        torch.testing.assert_close(response, reverse)
        torch.testing.assert_close(transmission, material(sample, wi, normal, normal)[1])
        self.assertTrue(((transmission >= 0) & (transmission <= 1)).all())
        self.assertTrue(torch.autograd.gradcheck(lambda c: material(c, wi, wo, normal)[0], (sample,)))
        back, _, _ = material(sample, wi, -normal, normal)
        torch.testing.assert_close(back, torch.zeros_like(back), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(material(sample, wi, torch.tensor([[1., 0., 0.]], dtype=torch.double), normal)[0]).all())

    def test_analytic_material_initialization_and_checkpoint(self):
        model = build_transport(dict(representation='neural_material', material_model='ggx',
                                     feature_dim=32, rank=5, width=16), 1.).double()
        g = Gaussians(12, torch.zeros(3), 1., device='cpu', geometry='2dgs').double()
        before = g.params['features'].detach().clone()
        model.initialize_material(g, reset_normal=False)
        torch.testing.assert_close(g.params['features'][:, 6:], before[:, 6:], rtol=0, atol=0)
        self.assertFalse(list(model.decoder.parameters()))
        restored = build_transport(dict(representation='neural_material', material_model='ggx',
                                        feature_dim=32, rank=5, width=16), 1.).double()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(self.shade(model), self.shade(restored), rtol=0, atol=0)

    def test_sdf_silhouette_rays_and_geometry_gradient(self):
        from sdf import SurfaceSDF, silhouette_rays, silhouette_logits
        field = SurfaceSDF(torch.zeros(3), 1.)
        with torch.no_grad():
            field.network[-1].weight.zero_()
            field.network[-1].bias.zero_()
        c2w = torch.eye(4); c2w[2, 3] = -2
        view = torch.eye(4); view[2, 3] = 2
        sample = {'alpha':torch.ones(64, 64, 1), 'c2w':c2w, 'viewmat':view,
                  'K':torch.tensor([[32., 0, 32], [0, 32, 32], [0, 0, 1]])}
        rays = silhouette_rays(sample, field)
        indices = rays['valid']
        distance = torch.linalg.cross(rays['origin'].expand(len(indices), -1),
                                       rays['direction'][indices]).norm(dim=-1)
        logits = silhouette_logits(field, rays, indices)
        away_from_boundary = (distance-.6).abs() > .001
        torch.testing.assert_close((logits > 0)[away_from_boundary],
                                   (distance < .6)[away_from_boundary])
        # A larger analytic sphere should drive the SDF bias down, expanding it.
        loss = F.binary_cross_entropy_with_logits(logits, (distance < .7).float())
        loss.backward()
        self.assertGreater(float(field.network[-1].bias.grad), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in field.parameters()))

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.g = Gaussians(17, torch.zeros(3), 1., 32, device='cpu').double()
        p = self.g.params
        self.receivers = dict(means=p['means'][:11], features=p['features'][:11],
                              base=p['base'][:11], visibility=torch.linspace(.1, 1, 11).double(),
                              normals=torch.tensor([0., 0., 1.], dtype=torch.double).expand(11, -1))
        self.receivers['material'] = {
            'diffuse': torch.full((11, 3), .3, dtype=torch.double),
            'f0': torch.full((11, 3), .04, dtype=torch.double),
            'alpha': torch.tensor([.1, .3], dtype=torch.double).expand(11, -1),
            'mix': torch.full((11, 1), .9, dtype=torch.double)}
        self.eye = torch.tensor([0., 1., 4.], dtype=torch.double)
        self.light = torch.tensor([2., 3., 4.], dtype=torch.double)
        self.intensity = torch.tensor([1., 2., 3.], dtype=torch.double)
        self.visibility = torch.linspace(.2, 1, 17).double()

    def model(self, name):
        torch.manual_seed(7)
        config = resolve_config(dict(representation=name, rank=5))
        return self.placed(build_transport(config, 1.)), config

    @staticmethod
    def placed(model):
        # Light-space methods rasterize their light pass with gsplat, which needs CUDA float32.
        return model.cuda().float() if getattr(model, 'light_space', False) else model.double()

    def light_space_inputs(self):
        g = Gaussians(17, torch.zeros(3), 1., 32, device='cuda')
        with torch.no_grad():
            for key, value in self.g.params.items():
                g.params[key].copy_(value.float())
        p = g.params
        receivers = dict(means=p['means'][:11], features=p['features'][:11], base=p['base'][:11],
                         visibility=self.receivers['visibility'].float().cuda(),
                         normals=self.receivers['normals'].float().cuda())
        return g, receivers

    def shade(self, model, active=True, intensity=None, receivers=None):
        intensity = self.intensity if intensity is None else intensity
        if getattr(model, 'light_space', False):
            if not hasattr(self, '_light_space_inputs'):
                self._light_space_inputs = self.light_space_inputs()
            g, light_receivers = self._light_space_inputs
            cuda = lambda value: value.float().cuda()
            return model(g, light_receivers if receivers is None else receivers, cuda(self.eye), cuda(self.light),
                         cuda(intensity), None, active)
        return model(self.g, self.receivers if receivers is None else receivers,
                     self.eye, self.light, intensity, self.visibility, active)

    @staticmethod
    def needs_unavailable_cuda(name):
        return getattr(METHODS[name], 'light_space', False) and not torch.cuda.is_available()

    def test_method_selection_and_config_roundtrip(self):
        for name in METHODS:
            with self.subTest(name=name):
                argv = ['--representation', name]
                if 'rank' in METHODS[name].cli_fields:
                    argv += ['--rank', '5']
                parser = argparse.ArgumentParser()
                add_method_arguments(parser, argv)
                config = resolve_config(vars(parser.parse_args(argv)))
                self.assertEqual(config['representation'], name)
                if self.needs_unavailable_cuda(name):
                    continue
                model = self.placed(build_transport(config, 1.))
                buffer = io.BytesIO()
                torch.save(dict(config=config, transport=model.state_dict()), buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=False)
                restored = self.placed(build_transport(saved['config'], 1.))
                restored.load_state_dict(saved['transport'], strict=True)
                torch.testing.assert_close(self.shade(model), self.shade(restored), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            build_transport(dict(representation='missing_method'), 1.)

    def test_retired_methods_rejected_by_cli_factory_and_manifest(self):
        import contextlib
        from pathlib import Path
        from make_validation_manifest import build_manifest
        retired = ('learned_anchor_exchange', 'distribution_material', 'local_transport',
                   'native_2dgs', 'gggs_core', 'gaussian_wrapping')
        for name in retired:
            with self.subTest(name=name):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    add_method_arguments(argparse.ArgumentParser(), ['--representation', name])
                with self.assertRaises(ValueError):
                    build_transport({'representation': name}, 1.)
                config = dict(name='retired_probe', python='python', data_root='/data', worker_gpus=[0],
                              train={'representation': name},
                              families={'family': {'scenes': ['scene'], 'train': {}}}, launch_environment={})
                with self.assertRaises(ValueError):
                    build_manifest(config, Path('/tmp/port-retired-method-probe'))

    def test_explicit_checkpoints_are_evaluated_after_training(self):
        from pathlib import Path
        from make_validation_manifest import build_manifest
        config = dict(name='probe',python='python',data_root='/data',worker_gpus=[0],
                      train={'steps':4,'save-steps':[3],'lr-decay-steps':3},
                      families={'family':{'scenes':['scene'],'train':{}}},launch_environment={})
        manifest = build_manifest(config,Path('/tmp/port-checkpoint-probe'))
        phases = manifest['jobs'][0]['steps']
        self.assertEqual([p['name'] for p in phases],['train','eval_000003','eval'])
        argv = phases[0]['argv']
        self.assertEqual(argv[argv.index('--save-steps')+1],'3')
        self.assertTrue(phases[1]['resultpath'].endswith('test_000003/metrics.json'))
        self.assertIn('/tmp/port-checkpoint-probe/family/scene/step_000003.pt',phases[1]['argv'])
        self.assertTrue(phases[2]['resultpath'].endswith('test/metrics.json'))
        config.update(eval_branch='sdf_volume', eval_limit=2, eval_resolution=512, eval_highlights=True)
        volume = build_manifest(config, Path('/tmp/port-volume-checkpoint-probe'))
        for phase in volume['jobs'][0]['steps'][1:]:
            self.assertIn('--sdf-volume', phase['argv'])
            self.assertEqual(phase['argv'][phase['argv'].index('--limit')+1], '2')
            self.assertEqual(phase['argv'][phase['argv'].index('--resolution')+1], '512')
            self.assertIn('--highlights', phase['argv'])
        self.assertTrue(volume['jobs'][0]['resultpath'].endswith('test_sdf/metrics.json'))

    def test_full_validation_and_pair_exports_do_not_inherit_fit_limit(self):
        from pathlib import Path
        from make_validation_manifest import build_manifest

        config = dict(name='surface_probe', python='python', data_root='/data', worker_gpus=[0, 1],
                      train={'representation':'surface_attention', 'steps':100000,
                             'save-steps':[25000, 50000]},
                      families={'family':{'scenes':['scene'], 'train':{}}}, launch_environment={},
                      eval_split='fit', eval_limit=16, eval_resolution=512, eval_highlights=True,
                      eval_save_all=True, eval_full_fit=True, eval_validation=True, eval_test=False)
        manifest = build_manifest(config, Path('/tmp/port-surface-validation-probe'))
        phases = manifest['jobs'][0]['steps']
        self.assertEqual([phase['name'] for phase in phases],
                         ['train', 'eval_025000', 'eval_050000', 'eval', 'eval_full_fit', 'eval_validation'])
        for phase in phases[1:]:
            argv = phase['argv']
            self.assertIn('--save-all', argv)
            self.assertIn('--highlights', argv)
            self.assertEqual(argv[argv.index('--resolution')+1], '512')
            self.assertNotEqual(argv[argv.index('--split')+1], 'test')
        for phase in phases[-2:]:
            self.assertNotIn('--limit', phase['argv'])
            self.assertTrue(any(value.endswith('/last.pt') for value in phase['argv']))
        self.assertEqual(phases[-1]['argv'][phases[-1]['argv'].index('--split')+1], 'validation')
        config.update(eval_split='validation', eval_limit=0, eval_full_fit=False)
        with self.assertRaisesRegex(ValueError, 'repeat.*validation'):
            build_manifest(config, Path('/tmp/port-duplicate-validation-probe'))


    def test_material_highlight_survives_transport_gate_and_codes_are_not_fit_by_indirect(self):
        model, _ = self.model('neural_material')
        incident, wi = model.point_light(self.receivers['means'], self.light, self.intensity,
                                         self.receivers['visibility'])
        wo = F.normalize(self.eye-self.receivers['means'],dim=-1)
        specular, _ = model.material_response(self.receivers,wi,wo)
        with torch.no_grad():
            model.exchange.bias.fill_(80.)
            model.raw_C.fill_(-80.)
        torch.testing.assert_close(self.shade(model), incident*specular, rtol=1e-12, atol=1e-12)
        model, _ = self.model('neural_material')
        with torch.no_grad():
            model.exchange.weight.normal_(std=.2)
        dark = dict(self.receivers,visibility=torch.zeros_like(self.receivers['visibility']))
        self.shade(model,receivers=dark).sum().backward()
        gradient=self.g.params['features'].grad
        torch.testing.assert_close(gradient[:,:9],torch.zeros_like(gradient[:,:9]),rtol=0,atol=0)
        self.assertGreater(float(gradient[:,9:].abs().sum()),0.)

    def test_zero_light_linearity_and_optimization(self):
        for name in METHODS:
            with self.subTest(name=name):
                if self.needs_unavailable_cuda(name):
                    continue
                model, _ = self.model(name)
                out = self.shade(model)
                torch.testing.assert_close(self.shade(model, intensity=self.intensity*3), out*3)
                torch.testing.assert_close(self.shade(model, intensity=self.intensity*0), torch.zeros_like(out), rtol=0, atol=0)
                parameters = list(model.parameters())
                optimizer = torch.optim.Adam(parameters, lr=.001)
                before = {k: v.detach().clone() for k, v in model.named_parameters()}
                for _ in range(3):
                    optimizer.zero_grad()
                    self.shade(model).square().mean().backward()
                    for key, parameter in model.named_parameters():
                        if not parameter.requires_grad:
                            self.assertIsNone(parameter.grad, key)
                            torch.testing.assert_close(parameter, before[key], rtol=0, atol=0)
                            continue
                        self.assertIsNotNone(parameter.grad, key)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                    optimizer.step()

    def test_attention_matches_explicit_cross_attention_and_source_permutation(self):
        from methods.base import SourceLight, ReceiverLight, direction_encoding
        model,_ = self.model('surface_attention')
        source = SourceLight(self.g.params['means'], self.g.params['features'],
                             torch.rand(17,3,dtype=torch.double),
                             F.normalize(torch.randn(17,3,dtype=torch.double),dim=-1),
                             quadrature_mass(self.g.params))
        receiver = ReceiverLight(self.receivers['means'],self.receivers['features'],
                                 torch.rand(11,3,dtype=torch.double),
                                 F.normalize(torch.randn(11,3,dtype=torch.double),dim=-1),
                                 torch.rand(11,3,dtype=torch.double))
        k,v,m = model.source_tokens(source)
        q = model.query(torch.cat((receiver.features,direction_encoding(receiver.xyz),
                                  direction_encoding(receiver.direction)),dim=-1))
        weights = (q@k.T/model.attention_dim**.5 + m.log()).softmax(-1)
        a = model.exchange(receiver.features).sigmoid()
        expected = (1-a)*receiver.incident*receiver.response+a*(weights@v)
        actual = model.exchange_radiance(source,receiver)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        permutation = torch.randperm(17)
        shuffled = SourceLight(*(getattr(source,key)[permutation] for key in
                                ('xyz','features','incident','direction','mass')))
        torch.testing.assert_close(model.exchange_radiance(shuffled,receiver),actual,rtol=1e-12,atol=1e-12)

    def test_cosine_attention_ignores_query_key_norm_and_keeps_light_linearity(self):
        config = resolve_config(dict(representation='surface_attention', attention_score='cosine'))
        model = build_transport(config, 1.).double()
        reference = self.shade(model)
        with torch.no_grad():
            for network in (model.query, model.key):
                network[-1].weight.mul_(1000)
                network[-1].bias.mul_(1000)
        torch.testing.assert_close(self.shade(model), reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(self.shade(model, intensity=self.intensity*3), reference*3)
        torch.testing.assert_close(self.shade(model, intensity=self.intensity*0), torch.zeros_like(reference))
        self.shade(model).sum().backward()
        for network in (model.query, model.key):
            self.assertTrue(torch.isfinite(network[0].weight.grad).all())
            self.assertGreater(float(network[0].weight.grad.abs().sum()), 0.)
        restored = build_transport(config, 1.).double()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(self.shade(restored), reference, rtol=1e-12, atol=1e-12)

    def test_attention_checkpoint_without_score_retains_dot_behavior(self):
        model, config = self.model('surface_attention')
        config.pop('attention_score')
        restored = build_transport(config, 1.).double()
        restored.load_state_dict(model.state_dict())
        self.assertEqual(restored.attention_score, 'dot')
        torch.testing.assert_close(self.shade(restored), self.shade(model), rtol=0, atol=0)


class NeuralMaterialTests(unittest.TestCase):
    def test_procedural_fresnel_and_layer_energy(self):
        from materials.procedural import (
            _fresnel_dielectric, _fresnel_conductor, enhanced_brdf, sample_materials, sample_directions,
        )
        torch.testing.assert_close(_fresnel_dielectric(torch.ones(1, 1), torch.full((1, 1), 1.5)),
                                   torch.full((1, 1), .04))
        rng = torch.Generator().manual_seed(8)
        params, _ = sample_materials(512, 'cpu', rng)
        wi, wo = sample_directions(512, 'cpu', rng, params=params)
        torch.testing.assert_close(wi.norm(dim=-1), torch.ones(512))
        torch.testing.assert_close(wo.norm(dim=-1), torch.ones(512))
        self.assertTrue(bool((wi[:, 2] > 0).all() and (wo[:, 2] > 0).all()))
        f, t, r = enhanced_brdf(params, wi, wo)
        self.assertTrue(bool(torch.isfinite(f).all() and (f >= 0).all()))
        self.assertTrue(bool((t >= 0).all() and (r >= 0).all()))
        self.assertTrue(bool((t + r <= 1 + 1e-5).all()))
        _, other_t, other_r = enhanced_brdf(params, wi, wo.flip(0))
        torch.testing.assert_close(t, other_t, rtol=0, atol=0)
        torch.testing.assert_close(r, other_r, rtol=0, atol=0)
        params[:, 10] = 1
        params[:, 11] = params[:, 13] = 0
        _, metal_t, _ = enhanced_brdf(params, wi, wo)
        torch.testing.assert_close(metal_t, torch.zeros_like(metal_t), rtol=0, atol=0)
        n, k = params[:, 1:4], params[:, 4:7]
        f0 = ((n-1).square()+k.square()) / ((n+1).square()+k.square())
        torch.testing.assert_close(_fresnel_conductor(torch.ones(512, 1), n, k), f0)

    def test_decoder_frozen_across_stages_but_latent_and_normal_differentiate(self):
        from training.schedule import set_training_stage
        model = build_transport(dict(representation='neural_material', rank=5), 1.)
        g = Gaussians(8, torch.zeros(3), 1., device='cpu', geometry='2dgs')
        optimizers = g.optimizers()
        frozen = {k: v.detach().clone() for k, v in model.decoder.named_parameters()}
        for step in (1, 3):
            set_training_stage(g, model, optimizers, step, 2)
            self.assertTrue(all(not p.requires_grad for p in model.decoder.parameters()))
        normal = torch.tensor([[.1, .1, 1.]]).expand(8, -1).clone().requires_grad_()
        wi = F.normalize(torch.tensor([[.3, .1, 1.]]).expand(8, -1), dim=-1)
        wo = F.normalize(torch.tensor([[0., .1, 1.]]).expand(8, -1), dim=-1)
        receivers = dict(features=g.params['features'], base=g.params['base'],
                         normals=F.normalize(normal, dim=-1))
        value = model.direct_response(receivers, g.params['means'], wi, wo)
        moved = model.direct_response(receivers, g.params['means'] + 100, wi, wo)
        torch.testing.assert_close(value, moved, rtol=0, atol=0)
        value.sum().backward()
        for gradient in (g.params['features'].grad[:, :6], normal.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
        for key, parameter in model.decoder.named_parameters():
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, frozen[key], rtol=0, atol=0)


class NeutralPeakComponentTests(unittest.TestCase):
    @staticmethod
    def image(points=(), height=48, width=160):
        image = torch.tensor([.1, .2, .4]).expand(height, width, 3).clone()
        for y, x in points:
            image[y, x] = .9
        return image

    def assert_peak_pixels(self, image, alpha, points):
        from evaluate import neutral_peak_mask
        expected = torch.zeros(image.shape[:2], dtype=torch.bool)
        for y, x in points:
            expected[y, x] = True
        torch.testing.assert_close(neutral_peak_mask(image, alpha), expected)

    def test_component_bucket_boundaries_and_eight_connectivity(self):
        from evaluate import neutral_peak_components
        components = [
            [(12, 12)],
            [(12+i, 40+i) for i in range(4)],  # One 8-connected component.
            [(12, 74), (11, 75), (12, 75), (13, 75), (12, 76)],
            [(12+y, 105+x) for y in range(4) for x in range(4)],
            [(12+y, 140+x) for y in range(4) for x in range(4)]+[(16, 140)],
        ]
        points = [point for component in components for point in component]
        target = self.image(points)
        alpha = torch.ones(*target.shape[:2], 1)
        self.assert_peak_pixels(target, alpha, points)
        row = neutral_peak_components(target, target, alpha)
        self.assertEqual(row['total_predicted_peak_pixels'], 43)
        for bucket, count, pixels in [('1_to_4', 2, 5), ('5_to_16', 2, 21), ('gt16', 1, 17)]:
            with self.subTest(bucket=bucket):
                raw, metric = row['buckets'][bucket]['sums'], row['buckets'][bucket]['metrics']
                self.assertEqual(raw['component_count'], count)
                self.assertEqual(raw['gt_peak_pixels'], pixels)
                self.assertEqual(raw['matched_gt_pixels_2px'], pixels)
                self.assertEqual(raw['components_with_at_least_one_hit'], count)
                self.assertEqual(metric['rgb_mae'], 0.)
                self.assertEqual(metric['contrast_ratio'], 1.)
                self.assertEqual(metric['pixel_recall_2px'], 1.)
                self.assertEqual(metric['component_any_hit_fraction'], 1.)

    def test_two_pixel_chebyshev_hit_and_component_any_hit(self):
        from evaluate import neutral_peak_components
        gt_points = [(20, x) for x in range(20, 24)]+[(20, 40)]
        # (22,25) reaches only (20,23), diagonally at radius two. Radius
        # three at (23,40) must not count as a hit on the singleton.
        pred_points = [(22, 25), (23, 40)]
        target, predicted = self.image(gt_points), self.image(pred_points)
        alpha = torch.ones(*target.shape[:2], 1)
        self.assert_peak_pixels(target, alpha, gt_points)
        self.assert_peak_pixels(predicted, alpha, pred_points)
        row = neutral_peak_components(predicted, target, alpha)
        metric = row['buckets']['1_to_4']['metrics']
        self.assertEqual(row['total_predicted_peak_pixels'], 2)
        self.assertEqual(metric['component_count'], 2)
        self.assertEqual(metric['gt_peak_pixels'], 5)
        self.assertEqual(metric['matched_gt_pixels_2px'], 1)
        self.assertEqual(metric['components_with_at_least_one_hit'], 1)
        self.assertEqual(metric['pixel_recall_2px'], .2)
        self.assertEqual(metric['component_any_hit_fraction'], .5)

    def test_empty_ground_truth_prediction_and_aggregate(self):
        from evaluate import aggregate_neutral_peak_components, neutral_peak_components
        empty, peak = self.image(), self.image([(20, 20)])
        alpha = torch.ones(*empty.shape[:2], 1)
        no_gt = neutral_peak_components(peak, empty, alpha)
        self.assertEqual(no_gt['total_predicted_peak_pixels'], 1)
        for row in [no_gt, neutral_peak_components(empty, empty, alpha),
                    aggregate_neutral_peak_components([])]:
            for bucket in row['buckets'].values():
                self.assertTrue(all(value == 0 for value in bucket['sums'].values()))
                for key in ('rgb_mae', 'target_contrast', 'predicted_contrast',
                            'contrast_ratio', 'pixel_recall_2px', 'component_any_hit_fraction'):
                    self.assertIsNone(bucket['metrics'][key])
        no_prediction = neutral_peak_components(empty, peak, alpha)
        self.assertEqual(no_prediction['total_predicted_peak_pixels'], 0)
        metric = no_prediction['buckets']['1_to_4']['metrics']
        self.assertEqual(metric['component_count'], 1)
        self.assertEqual(metric['gt_peak_pixels'], 1)
        self.assertEqual(metric['matched_gt_pixels_2px'], 0)
        self.assertEqual(metric['components_with_at_least_one_hit'], 0)
        self.assertEqual(metric['pixel_recall_2px'], 0.)
        self.assertEqual(metric['component_any_hit_fraction'], 0.)
        self.assertAlmostEqual(metric['rgb_mae'], 2./3., places=6)

    def test_raw_sum_pooling_weights_pixels_and_components_separately(self):
        from evaluate import aggregate_neutral_peak_components, neutral_peak_components
        first = self.image([(20, 20)])
        second = self.image([(20, 20), (20, 21), (21, 20), (21, 21)])
        alpha = torch.ones(*first.shape[:2], 1)
        rows = [neutral_peak_components(self.image(), first, alpha),
                neutral_peak_components(second, second, alpha)]
        pooled = aggregate_neutral_peak_components(rows)
        raw = pooled['buckets']['1_to_4']['sums']
        metric = pooled['buckets']['1_to_4']['metrics']
        self.assertEqual(pooled['total_predicted_peak_pixels'], 4)
        for key in raw:
            self.assertEqual(raw[key], sum(row['buckets']['1_to_4']['sums'][key] for row in rows))
        self.assertEqual(metric['component_count'], 2)
        self.assertEqual(metric['gt_peak_pixels'], 5)
        self.assertAlmostEqual(metric['rgb_mae'], 2./15., places=6)
        self.assertEqual(metric['pixel_recall_2px'], .8)
        self.assertEqual(metric['component_any_hit_fraction'], .5)
        self.assertEqual(metric['contrast_ratio'], raw['predicted_contrast_sum']/raw['target_contrast_sum'])
        frame_average = sum(row['buckets']['1_to_4']['metrics']['rgb_mae'] for row in rows)/2
        self.assertGreater(abs(frame_average-metric['rgb_mae']), .1)

    def test_alpha_interior_excludes_border_and_partial_foreground(self):
        from evaluate import neutral_peak_components
        points = [(1, 10), (12, 12), (24, 24), (36, 50)]
        target = self.image(points)
        alpha = torch.ones(*target.shape[:2], 1)
        alpha[12, 14, 0] = 0.  # Radius-two alpha hole rejects this peak.
        alpha[24, 27, 0] = 0.  # Radius-three hole leaves this peak intact.
        alpha[36, 50, 0] = .9  # Foreground threshold is strictly greater.
        self.assert_peak_pixels(target, alpha, [(24, 24)])
        row = neutral_peak_components(target, target, alpha)
        self.assertEqual(row['total_predicted_peak_pixels'], 1)
        self.assertEqual(row['buckets']['1_to_4']['metrics']['component_count'], 1)
        self.assertEqual(row['buckets']['1_to_4']['metrics']['gt_peak_pixels'], 1)


class AngularCueTests(unittest.TestCase):
    @staticmethod
    def inputs(points=((20, 20),), width=80):
        target = NeutralPeakComponentTests.image(points, height=48, width=width)
        predicted = NeutralPeakComponentTests.image(height=48, width=width)
        alpha = torch.ones(48, width)
        angles = torch.full((48, width), 8.*torch.pi/180, dtype=torch.float64)
        return predicted, target, alpha, alpha.clone(), {'geometry': angles, 'material': angles.clone()}

    def test_kernel_unit_peak_and_radian_bandwidth(self):
        import math
        from angular_cues import angular_kernel_bank, NARROW_TAU_DEGREES, WIDE_TAU_DEGREES
        for bank, widths in [('narrow', NARROW_TAU_DEGREES), ('wide', WIDE_TAU_DEGREES)]:
            result = torch.from_numpy(angular_kernel_bank(torch.tensor([0., math.radians(widths[0])],
                                                                       dtype=torch.float64), bank))
            torch.testing.assert_close(result[0], torch.ones(8, dtype=torch.float64), rtol=0, atol=0)
            self.assertAlmostEqual(float(result[1, 0]), math.exp(-.5), places=12)
            self.assertTrue(bool((result[1, 1:] > result[1, :-1]).all()))
        self.assertEqual((NARROW_TAU_DEGREES[0], NARROW_TAU_DEGREES[-1]), (2., 32.))
        self.assertEqual((WIDE_TAU_DEGREES[0], WIDE_TAU_DEGREES[-1]), (8., 128.))

    def test_centered_displaced_flat_and_ambiguous_cues(self):
        from angular_cues import analyze_angular_cues
        for kind in ('centered', 'displaced', 'flat', 'tied'):
            with self.subTest(kind=kind):
                args = self.inputs()
                angles = args[-1]['geometry']
                if kind in ('centered', 'tied'):
                    angles[20, 20] = 0.
                if kind in ('displaced', 'tied'):
                    angles[20, 24] = 0.
                report = analyze_angular_cues(*args)
                cue = report['components'][0]['normals']['geometry']
                location = cue['localization']
                self.assertTrue(cue['eligible'])
                self.assertEqual(cue['qualifies'], kind == 'centered')
                self.assertEqual(location['reliable'], kind == 'centered')
                if kind == 'centered':
                    self.assertEqual(location['best_distance_px'], 0)
                    self.assertEqual(location['worst_distance_px'], 0)
                    self.assertEqual(location['tie_count'], 1)
                    self.assertGreater(cue['banks']['narrow']['best_delta'], .99)
                    self.assertGreater(cue['banks']['narrow']['best_delta'] -
                                       cue['banks']['wide']['best_delta'], .5)
                elif kind == 'displaced':
                    self.assertEqual(location['best_distance_px'], 4)
                    self.assertEqual(location['worst_distance_px'], 4)
                elif kind == 'flat':
                    self.assertTrue(location['plateau'])
                else:
                    self.assertEqual(location['tie_count'], 2)
                    self.assertEqual(location['best_distance_px'], 0)
                    self.assertEqual(location['worst_distance_px'], 4)
                    self.assertTrue(location['tie_spans_core_and_ring'])

    def test_component_connectivity_and_ring_exclusions(self):
        import math
        from angular_cues import analyze_angular_cues
        points = [(20+i, 20+i) for i in range(4)]+[(20, 26)]
        args = self.inputs(points)
        predicted, target, target_alpha, gs_alpha, maps = args
        target_alpha[15, 15] = 0.  # Beyond the canonical peak's 5px interior check.
        gs_alpha[15, 16] = 0.
        for angles in maps.values():
            for y, x in points:
                angles[y, x] = 0.
            angles[15, 15] = 0.
            angles[15, 16] = float('nan')
        report = analyze_angular_cues(*args)
        self.assertEqual(report['counts']['gt_small_peak_components'], 2)
        self.assertEqual(report['counts']['gt_small_peak_pixels'], 5)
        row = next(row for row in report['components'] if row['area_pixels'] == 4)
        cue = row['normals']['geometry']
        self.assertEqual(cue['valid_core_pixels'], 4)
        # Four diagonal 11x11 squares have union184: exclude four own
        # peaks, one foreign GT peak, one alpha hole and one GS hole.
        self.assertEqual(cue['valid_ring_pixels'], 177)
        self.assertAlmostEqual(cue['banks']['narrow']['core_mean'][0], 1., places=12)
        self.assertAlmostEqual(cue['banks']['narrow']['ring_mean'][0], math.exp(-8.), places=12)

    def test_hit_groups_use_two_pixel_chebyshev_tolerance(self):
        from angular_cues import analyze_angular_cues
        gt_points = [(20, x) for x in range(20, 24)]+[(20, 40), (20, 55)]
        args = list(self.inputs(gt_points))
        args[0] = NeutralPeakComponentTests.image([(22, 25), (22, 42), (23, 55)], height=48, width=80)
        report = analyze_angular_cues(*args)
        self.assertEqual(report['counts']['gt_small_peak_components'], 3)
        self.assertEqual(report['counts']['gt_small_peak_pixels'], 6)
        self.assertEqual(report['counts']['matched_gt_pixels_2px'], 2)
        self.assertEqual(report['counts']['components_with_at_least_one_hit'], 2)
        groups = {row['bbox_xywh'][0]: row['hit_group'] for row in report['components']}
        self.assertEqual(groups, {20: 'partial', 40: 'all_hit', 55: 'missed'})

    def test_missing_core_empty_ring_and_low_alpha_support(self):
        from angular_cues import analyze_angular_cues
        args = self.inputs([(20, 20), (20, 40), (20, 60)])
        gs_alpha, maps = args[3:]
        gs_alpha[20, 20] = 0.
        gs_alpha[20, 40] = .4  # Positive GS support remains eligible, reported as low alpha.
        gs_alpha[15:26, 55:66] = 0.
        gs_alpha[20, 60] = 1.
        for angles in maps.values():
            angles[20, 40] = angles[20, 60] = 0.
            angles[gs_alpha == 0] = float('nan')
        report = analyze_angular_cues(*args)
        rows = {row['bbox_xywh'][0]: row for row in report['components']}
        self.assertEqual(rows[20]['coverage']['uncovered_core_pixels'], 1)
        self.assertEqual(rows[40]['coverage']['low_alpha_core_pixels'], 1)
        self.assertFalse(rows[20]['normals']['geometry']['eligible'])
        self.assertTrue(rows[40]['normals']['geometry']['eligible'])
        self.assertTrue(rows[40]['normals']['geometry']['qualifies'])
        self.assertFalse(rows[60]['normals']['geometry']['eligible'])
        self.assertEqual(rows[60]['normals']['geometry']['valid_ring_pixels'], 0)
        self.assertEqual(sum(row['normals']['geometry']['qualifies'] for row in rows.values()), 1)
        self.assertEqual(report['screen']['missed_components_denominator'], 3)
        self.assertEqual(report['screen']['unsupported_components'], 2)
        self.assertEqual(report['screen']['qualifying_fraction'], 1/3)
        self.assertFalse(report['screen']['numerical_screen_passes'])

    def test_merge_weights_components_pixels_and_unsupported_denominator(self):
        import math
        from angular_cues import analyze_angular_cues, aggregate_angular_cues
        single = self.inputs()
        for angles in single[-1].values():
            angles[20, 20] = 0.
        four = self.inputs([(20, 20), (20, 21), (21, 20), (21, 21)])
        empty = self.inputs([])
        unsupported = self.inputs()
        unsupported[3][20, 20] = 0.
        for angles in unsupported[-1].values():
            angles[20, 20] = float('nan')
        reports = [dict(analyze_angular_cues(*args), frame_index=index)
                   for index, args in enumerate((single, four, empty, unsupported))]
        combined = aggregate_angular_cues(reports)
        self.assertEqual(combined['frame_count'], 4)
        self.assertEqual(combined['counts']['gt_small_peak_components'], 3)
        self.assertEqual(combined['counts']['gt_small_peak_pixels'], 6)
        self.assertEqual([row['frame_index'] for row in combined['components']], [0, 1, 3])
        summary = combined['aggregate']['missed']['normals']['geometry']
        self.assertEqual((summary['eligible_components'], summary['eligible_pixels']), (2, 5))
        means = summary['eligible_bank_means']['narrow']['core_mean']
        self.assertAlmostEqual(means['component_weighted'][0], (1+math.exp(-8.))/2, places=12)
        self.assertAlmostEqual(means['pixel_weighted'][0], (1+4*math.exp(-8.))/5, places=12)
        self.assertEqual(combined['screen']['missed_components_denominator'], 3)
        self.assertEqual(combined['screen']['qualifying_components'], 1)
        self.assertEqual(combined['screen']['qualifying_fraction'], 1/3)
        self.assertFalse(combined['screen']['numerical_screen_passes'])
        empty_group = combined['aggregate']['all_hit']['normals']['geometry']
        self.assertEqual(empty_group['components'], 0)
        self.assertIsNone(empty_group['eligible_metric_distributions']['best_narrow_delta']['pixel_weighted']['mean'])
        self.assertIsNone(empty_group['eligible_bank_means']['narrow']['core_mean']['component_weighted'])
        empty_merged = aggregate_angular_cues([reports[2]])
        self.assertEqual(empty_merged['screen']['missed_components_denominator'], 0)
        self.assertIsNone(empty_merged['screen']['qualifying_fraction'])
        self.assertFalse(empty_merged['screen']['numerical_screen_passes'])


class ResidualFramePoolTests(unittest.TestCase):
    def test_default_and_explicit_pools_preserve_source_membership(self):
        from training.residual import residual_frame_pool
        fit = [9, 2, 7, 15]
        frames = [15, 2]
        for enabled in (False, True):
            self.assertEqual(residual_frame_pool(fit, None, enabled), [9, 2, 7, 15])
        pool = residual_frame_pool(fit, frames, True)
        self.assertEqual(pool, [15, 2])
        # A stage subset does not redefine checkpoint fit membership.
        self.assertEqual(fit, [9, 2, 7, 15])
        self.assertEqual(frames, [15, 2])

    def test_invalid_residual_frame_pools_are_rejected(self):
        from training.residual import residual_frame_pool
        fit = [9, 2, 7, 15]
        for frames, enabled in [([2, 2], True), ([15, 99], True), ([2], False), ([], True)]:
            with self.subTest(frames=frames, enabled=enabled):
                with self.assertRaises(ValueError):
                    residual_frame_pool(fit, frames, enabled)
                self.assertEqual(fit, [9, 2, 7, 15])


class RadianceResidualTests(unittest.TestCase):
    def setUp(self):
        from materials.radiance_residual import RadianceResidual
        from methods.neural_material import NeuralMaterialTransport
        torch.manual_seed(19)
        self.head = RadianceResidual(torch.zeros(3), 2.)
        self.transport = NeuralMaterialTransport(rank=4)
        self.transport.light_scale.requires_grad_()
        self.covered = torch.tensor([[True, False, True, True], [False, True, False, True]])
        self.receivers = {
            'means':torch.tensor([[-.4, 0, 0], [-.2, .3, 0], [0, .1, .2], [.4, 0, .1], [.2, -.2, .1]],
                                 requires_grad=True),
            'normals':torch.tensor([[0., 0., 1.]]).repeat(5, 1).requires_grad_(),
            'features':(torch.randn(5, 32)*.2).requires_grad_()}
        self.sample = {'c2w':torch.eye(4, requires_grad=True),
                       'light_pos':torch.tensor([0., 4., 3.], requires_grad=True),
                       'light_intensity':torch.tensor([1., 2., 3.], requires_grad=True)}
        self.base = torch.full((2, 4, 3), .3, requires_grad=True)
        self.alpha = (self.covered[..., None].float()*.75).requires_grad_()

    def apply(self, indices=None, base=None, sample=None):
        from renderer import apply_radiance_residual
        return apply_radiance_residual(self.base if base is None else base, self.alpha,
            self.receivers, self.covered, self.transport, self.sample if sample is None else sample,
            self.head, indices)

    def test_zero_render_preservation_and_gradient_isolation(self):
        from evaluate import observation_image
        for source in ('geometry', 'material'):
            self.head.normal_source = source
            seen = []
            hook = self.head.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[1]))
            foreground, alpha, stats = self.apply()
            hook.remove()
            expected_normal = (self.receivers['normals'] if source == 'geometry'
                               else self.transport.material_normal(self.receivers))
            torch.testing.assert_close(seen[0], expected_normal, rtol=0, atol=0)
            self.assertFalse(seen[0].requires_grad)
            torch.testing.assert_close(foreground, self.base, rtol=0, atol=0)
            torch.testing.assert_close(alpha, self.alpha, rtol=0, atol=0)
            before = self.base*self.alpha+.2*(1-self.alpha)
            after = foreground*alpha+.2*(1-alpha)
            torch.testing.assert_close(after, before, rtol=0, atol=0)
            torch.testing.assert_close(observation_image(after, 2.2, alpha, .2),
                                       observation_image(before, 2.2, self.alpha, .2), rtol=0, atol=0)
            self.assertEqual(float(stats['delta_abs_mean']), 0.)
        after.sum().backward()
        self.assertGreater(float(self.head.network[-1].weight.grad.abs().sum()), 0.)
        for value in [self.base, self.alpha, self.transport.light_scale,
                      *self.receivers.values(), *self.sample.values(), *self.transport.parameters()]:
            self.assertIsNone(value.grad)

    def test_signed_clamp_and_positive_light_homogeneity(self):
        with torch.no_grad():
            self.head.network[-1].bias.copy_(torch.tensor([.1, -100., 100.]))
        foreground, alpha, stats = self.apply()
        self.assertTrue((foreground[self.covered][:, 1] == 0).all())
        self.assertGreater(float(foreground.max()), 1.)
        self.assertAlmostEqual(float(stats['clamp_fraction']), 1/3, places=6)
        brighter, _, _ = self.apply(base=3*self.base,
            sample=dict(self.sample, light_intensity=3*self.sample['light_intensity']))
        torch.testing.assert_close(brighter, 3*foreground)
        zero, _, _ = self.apply(base=torch.zeros_like(self.base),
            sample=dict(self.sample, light_intensity=torch.zeros(3)))
        torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
        torch.testing.assert_close(alpha, self.alpha, rtol=0, atol=0)
        torch.testing.assert_close(self.base, torch.full_like(self.base, .3), rtol=0, atol=0)

    def test_sparse_queries_keep_repeated_loss_weights_and_empty_support(self):
        from renderer import apply_radiance_residual
        ids = torch.tensor([0, 0, 1, 2, 7])
        calls = []
        hook = self.head.register_forward_pre_hook(lambda module, inputs: calls.append(len(inputs[0])))
        sparse, _, stats = self.apply(ids)
        hook.remove()
        self.assertEqual(calls, [3])
        self.assertEqual(stats['queried_pixels'], 3)
        self.assertEqual(int(stats['requested_covered_pixels']), 4)
        self.assertEqual(stats['requested_pixels'], 5)
        parameter = self.head.network[-1].weight
        sparse_grad = torch.autograd.grad(sparse.reshape(-1, 3)[ids].sum(), parameter)[0]
        # Pixel IDs 0, 0, 2, 7 map to these packed receivers; uncovered ID 1 contributes no correction.
        rows = torch.tensor([0, 0, 1, 4])
        direct = self.head(self.receivers['means'][rows], self.receivers['normals'][rows],
                           self.sample['c2w'][:3, 3], self.sample['light_pos'],
                           self.sample['light_intensity'], self.transport.light_scale)
        torch.testing.assert_close(sparse_grad, torch.autograd.grad(direct.sum(), parameter)[0])
        with torch.no_grad():
            self.head.network[-1].weight.normal_(std=.01)
        sparse, _, _ = self.apply(ids)
        full, _, _ = self.apply()
        torch.testing.assert_close(sparse.reshape(-1, 3)[ids], full.reshape(-1, 3)[ids])
        empty, _, stats = self.apply(torch.tensor([1, 4, 6]))
        self.assertEqual(stats['queried_pixels'], 0)
        torch.testing.assert_close(empty, self.base, rtol=0, atol=0)
        empty.sum().backward()
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)
        _, _, stats = apply_radiance_residual(self.base, self.alpha,
            {key:value[:0] for key,value in self.receivers.items()}, torch.zeros_like(self.covered),
            self.transport, self.sample, self.head)
        self.assertEqual(stats['covered_pixels'], 0)
        self.assertEqual(float(stats['clamp_fraction']), 0.)

    def test_full_query_chunks_are_bounded(self):
        from renderer import apply_radiance_residual
        receivers = {key:value[:1].expand(4097, -1) for key,value in self.receivers.items()}
        calls = []
        hook = self.head.register_forward_pre_hook(lambda module, inputs: calls.append(len(inputs[0])))
        with torch.no_grad():
            self.head.network[-1].bias.fill_(.1)
            foreground, _, stats = apply_radiance_residual(torch.full((1, 4097, 3), .3),
                torch.ones(1, 4097, 1), receivers, torch.ones(1, 4097, dtype=torch.bool),
                self.transport, self.sample, self.head)
        hook.remove()
        self.assertEqual(calls, [4096, 1])
        self.assertEqual(stats['queried_pixels'], 4097)
        torch.testing.assert_close(foreground, foreground[:, :1].expand_as(foreground))

    def test_strict_state_round_trip(self):
        from materials.radiance_residual import RadianceResidual
        with torch.no_grad():
            self.head.network[-1].weight.normal_(std=.01)
            self.head.detail.projection.weight.normal_(std=.05)
        reference = self.apply()[0]
        buffer = io.BytesIO()
        torch.save(self.head.state_dict(), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = RadianceResidual(torch.ones(3), 1.)
        restored.load_state_dict(saved, strict=True)
        self.head = restored
        torch.testing.assert_close(self.apply()[0], reference, rtol=0, atol=0)
        del saved['network.6.bias']
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(saved, strict=True)


class ResidualInteractionTests(unittest.TestCase):
    setUp = RadianceResidualTests.setUp
    apply = RadianceResidualTests.apply

    def make_head(self, mode):
        from materials.radiance_residual import RadianceResidual
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            return RadianceResidual(torch.zeros(3), 2., interaction=mode)

    def inputs(self):
        return (self.receivers['means'], self.receivers['normals'], self.sample['c2w'][:3, 3],
                self.sample['light_pos'], self.sample['light_intensity'], self.transport.light_scale)

    def test_shared_initial_state_zero_image_and_legacy_keys(self):
        add, mul, old = [self.make_head(mode) for mode in ('add', 'multiply', 'none')]
        self.assertEqual(add.state_dict().keys(), mul.state_dict().keys())
        for key, value in add.state_dict().items():
            torch.testing.assert_close(value, mul.state_dict()[key], rtol=0, atol=0)
            if key in old.state_dict():
                torch.testing.assert_close(value, old.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in add.parameters()) -
                         sum(p.numel() for p in old.parameters()), 3504)
        old.load_state_dict(self.head.state_dict(), strict=True)
        for head in (add, mul):
            self.head = head
            foreground, alpha, stats = self.apply()
            torch.testing.assert_close(foreground, self.base, rtol=0, atol=0)
            torch.testing.assert_close(alpha, self.alpha, rtol=0, atol=0)
            self.assertGreater(float(stats['interaction_output_square_mean']), 0.)
            self.assertEqual(float(stats['delta_abs_mean']), 0.)
        with self.assertRaises(ValueError):
            self.make_head('invalid')
        with self.assertRaises(RuntimeError):
            old.load_state_dict(add.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            add.load_state_dict(old.state_dict(), strict=True)

    def test_addition_absorbs_into_existing_linear_maps(self):
        add, old = self.make_head('add'), self.make_head('none')
        with torch.no_grad():
            add.network[-1].weight.normal_(std=.1)
            old.load_state_dict({k:v for k,v in add.state_dict().items()
                                 if k in old.state_dict()}, strict=True)
            old.network[0].weight[:, 27:] += (add.interaction_projection.weight @
                                             add.interaction_angular.weight)
            old.detail.projection.weight += (add.interaction_projection.weight @
                                              add.interaction_spatial.weight)
        torch.testing.assert_close(add(*self.inputs()), old(*self.inputs()), rtol=2e-5, atol=1e-8)

    def test_new_factor_gradients_detachment_and_light_scaling(self):
        for mode in ('add', 'multiply'):
            with self.subTest(mode=mode):
                self.head = self.make_head(mode)
                with torch.no_grad():
                    self.head.network[-1].weight.normal_(std=.1)
                result = self.head(*self.inputs())
                changed = list(self.inputs())
                changed[4] = changed[4]*3
                torch.testing.assert_close(self.head(*changed), result*3)
                result.square().mean().backward()
                for layer in (self.head.interaction_spatial, self.head.interaction_angular,
                              self.head.interaction_projection, self.head.detail.grid):
                    for parameter in layer.parameters():
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(float(parameter.grad.abs().sum()), 0.)
                for value in [*self.receivers.values(), *self.sample.values(), self.transport.light_scale]:
                    self.assertIsNone(value.grad)
                restored = self.make_head(mode)
                restored.load_state_dict(self.head.state_dict(), strict=True)
                torch.testing.assert_close(restored(*self.inputs()), result, rtol=0, atol=0)

    def test_diagnostics_reuse_grid_and_pool_unique_queries(self):
        from renderer import apply_radiance_residual
        for mode in ('add', 'multiply'):
            self.head = self.make_head(mode)
            count = []
            hook = self.head.detail.grid.register_forward_hook(lambda m, a, o: count.append(len(a[0])))
            output, direct = self.head(*self.inputs(), return_stats=True)
            hook.remove()
            self.assertEqual(count, [5])
            _, _, pooled = self.apply()
            for key, value in direct.items():
                torch.testing.assert_close(pooled[key], value)
                self.assertFalse(value.requires_grad)
            _, _, empty = self.apply(torch.tensor([1, 4, 6]))
            for key in direct:
                self.assertEqual(float(empty[key]), 0.)
            receivers = {key:value[:1].expand(4097, -1) for key,value in self.receivers.items()}
            receivers = {key:torch.cat((value[:4096], self.receivers[key][-1:]), 0)
                         for key,value in receivers.items()}
            with torch.no_grad():
                _, _, chunked = apply_radiance_residual(torch.full((1, 4097, 3), .3),
                    torch.ones(1, 4097, 1), receivers, torch.ones(1, 4097, dtype=torch.bool),
                    self.transport, self.sample, self.head)
                inputs = list(self.inputs())
                inputs[:2] = [receivers['means'], receivers['normals']]
                _, expected = self.head(*inputs, return_stats=True)
            for key, value in expected.items():
                torch.testing.assert_close(chunked[key], value, rtol=1e-5, atol=1e-12)


class MovableAngularCenterTests(unittest.TestCase):
    setUp = RadianceResidualTests.setUp
    apply = RadianceResidualTests.apply
    inputs = ResidualInteractionTests.inputs

    def make_head(self, bank):
        from materials.radiance_residual import RadianceResidual
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37)
            return RadianceResidual(torch.zeros(3), 2., interaction='multiply', angular_bank=bank)

    def test_shared_parameters_zero_image_and_legacy(self):
        wide, narrow, old = [self.make_head(bank) for bank in ('wide','narrow','none')]
        self.assertEqual(sum(p.numel() for p in wide.parameters()), 2200784)
        self.assertEqual(sum(p.numel() for p in wide.parameters())-sum(p.numel() for p in old.parameters()),2787)
        self.assertEqual(wide.state_dict().keys(),narrow.state_dict().keys())
        for key,value in wide.state_dict().items():
            if key != 'angular_scales':
                torch.testing.assert_close(value,narrow.state_dict()[key],rtol=0,atol=0)
            if key in old.state_dict():
                torch.testing.assert_close(value,old.state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(wide.angular_scales,4*narrow.angular_scales)
        for head in (wide,narrow):
            self.head=head
            image,alpha,stats=self.apply()
            torch.testing.assert_close(image,self.base,rtol=0,atol=0)
            torch.testing.assert_close(alpha,self.alpha,rtol=0,atol=0)
            self.assertEqual(float(stats['delta_abs_mean']),0.)
            self.assertEqual(float(stats['center_offset_square_mean']),0.)
            self.assertGreater(float(stats['center_raw_norm_min']),.99)
        with self.assertRaises(RuntimeError):
            old.load_state_dict(wide.state_dict(),strict=True)
        from materials.radiance_residual import RadianceResidual
        for options in ({'interaction':'add','normal_source':'geometry'},
                        {'interaction':'multiply','normal_source':'material'}):
            with self.assertRaises(ValueError):
                RadianceResidual(torch.zeros(3),2.,angular_bank='narrow',**options)

    def test_spherical_gaussian_unit_peak_poles_and_width(self):
        from materials.radiance_residual import spherical_gaussian_kernel
        scales = self.make_head('narrow').angular_scales.double()
        for sign in (1.,-1.):
            center=torch.tensor([[0.,0.,sign]],dtype=torch.float64,requires_grad=True)
            half=torch.tensor([[0.,0.,1.]],dtype=torch.float64,requires_grad=True)
            values=spherical_gaussian_kernel(center,half,scales)
            expected=torch.ones_like(scales) if sign==1 else torch.exp(-2/scales.square())
            torch.testing.assert_close(values[0],expected)
            values.sum().backward()
            self.assertTrue(torch.isfinite(center.grad).all())
            self.assertTrue(torch.isfinite(half.grad).all())
        theta=torch.tensor(.2,dtype=torch.float64)
        center=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
        half=torch.stack((theta.sin(),theta.new_zeros(()),theta.cos()))[None]
        small=spherical_gaussian_kernel(center,half,scales)
        broad=spherical_gaussian_kernel(center,half,4*scales)
        self.assertTrue((small<broad).all())
        torch.testing.assert_close(small[0],torch.exp((theta.cos()-1)/scales.square()))

    def test_center_learning_input_isolation_and_light_homogeneity(self):
        for bank in ('wide','narrow'):
            self.head=self.make_head(bank)
            # Start with a nonzero RGB map so this test addresses D's delayed
            # zero-final-layer learning, independent of the known RGB delay.
            with torch.no_grad(): self.head.network[-1].weight.normal_(std=.1)
            optimizer=torch.optim.Adam(self.head.parameters(),lr=.001,eps=1e-15)
            initial={key:value.clone() for key,value in self.head.center_network.state_dict().items()}
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                prediction=self.head(*self.inputs())
                (prediction-.05).square().mean().backward()
                for p in self.head.parameters():
                    if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
                if step==0:
                    torch.testing.assert_close(self.head.center_network[0].weight.grad,
                                               torch.zeros_like(self.head.center_network[0].weight))
                optimizer.step()
            self.assertTrue(all(not torch.equal(value,initial[key])
                                for key,value in self.head.center_network.state_dict().items()))
            self.assertGreater(float(self.head.center_network[0].weight.grad.abs().sum()),0.)
            prediction,stats=self.head(*self.inputs(),return_stats=True)
            self.assertGreater(float(stats['center_offset_square_mean']),0.)
            self.assertGreater(float(stats['center_rotation_deg_mean']),0.)
            changed=list(self.inputs());changed[4]=changed[4]*3
            torch.testing.assert_close(self.head(*changed),prediction*3)
            for value in [*self.receivers.values(),*self.sample.values(),self.transport.light_scale]:
                self.assertIsNone(value.grad)
            restored=self.make_head(bank);restored.load_state_dict(self.head.state_dict(),strict=True)
            torch.testing.assert_close(restored(*self.inputs()),prediction,rtol=0,atol=0)

    def test_query_grid_once_and_true_min_pooling(self):
        from renderer import apply_radiance_residual
        from evaluate import aggregate_residual_stats
        self.head=self.make_head('narrow')
        with torch.no_grad():
            self.head.center_network[-1].bias.copy_(torch.tensor([.1,.2,-.5]))
        calls=[]
        hook=self.head.detail.grid.register_forward_hook(lambda m,args,out:calls.append(len(args[0])))
        _,direct=self.head(*self.inputs(),return_stats=True)
        hook.remove();self.assertEqual(calls,[5])
        _,_,empty=self.apply(torch.tensor([1,4,6]))
        self.assertEqual(empty['queried_pixels'],0)
        self.assertEqual(float(empty['center_raw_norm_min']),0.)
        # Make the final one-pixel chunk have a distinctly smaller raw norm.
        receivers={key:value[:1].expand(4097,-1).clone() for key,value in self.receivers.items()}
        receivers['normals'][-1]=torch.tensor([0.,0.,.4])
        with torch.no_grad():
            _,_,chunked=apply_radiance_residual(torch.full((1,4097,3),.3),
                torch.ones(1,4097,1),receivers,torch.ones(1,4097,dtype=torch.bool),
                self.transport,self.sample,self.head)
            inputs=list(self.inputs());inputs[:2]=[receivers['means'],receivers['normals']]
            _,expected=self.head(*inputs,return_stats=True)
        for key,value in expected.items():
            torch.testing.assert_close(chunked[key],value,rtol=1e-5,atol=1e-9)
        rows=[{'queried_pixels':3,'covered_pixels':4,'requested_pixels':8,'requested_covered_pixels':5,
               'delta_abs_mean':2.,'center_raw_norm_min':.8},
              {'queried_pixels':1,'covered_pixels':2,'requested_pixels':8,'requested_covered_pixels':1,
               'delta_abs_mean':6.,'center_raw_norm_min':.2},
              {'queried_pixels':0,'covered_pixels':2,'requested_pixels':8,'requested_covered_pixels':0,
               'delta_abs_mean':0.,'center_raw_norm_min':0.}]
        pooled=aggregate_residual_stats(rows)
        self.assertEqual(pooled['queried_pixels'],4)
        self.assertEqual(pooled['delta_abs_mean'],3.)
        self.assertEqual(pooled['center_raw_norm_min'],.2)
        self.assertEqual(aggregate_residual_stats(rows[-1:])['center_raw_norm_min'],0.)


class SurfaceTests(unittest.TestCase):
    def test_normal_residual_zero_initialization_and_gradient(self):
        from materials.normal_field import NormalResidualField
        from methods.neural_material import NeuralMaterialTransport
        torch.manual_seed(29)
        field = NormalResidualField(torch.zeros(3), 1.)
        points = torch.rand(32,3)*.5
        receivers = {'normals':F.normalize(torch.randn(32,3),dim=-1), 'features':torch.randn(32,32)*.1}
        original = NeuralMaterialTransport.material_normal(receivers)
        corrected = dict(receivers, normal_residual=field(points))
        actual = NeuralMaterialTransport.material_normal(corrected)
        torch.testing.assert_close(original, actual, rtol=0, atol=0)
        target = F.normalize(torch.tensor([.2,.3,1.]),dim=0)
        optimizer = torch.optim.Adam(field.parameters(),lr=.001)
        (1-(actual*target).sum(-1)).mean().backward()
        grads = [p.grad for p in field.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        optimizer.step()
        learned = NeuralMaterialTransport.material_normal(dict(receivers,normal_residual=field(points)))
        self.assertGreater(float((learned-original).abs().max()),0.)
        torch.testing.assert_close(learned.norm(dim=-1),torch.ones(32))
        self.assertTrue(bool(((learned*receivers['normals']).sum(-1)>1/(1+.75)**.5).all()))

    def test_sdf_primitive_selection_rejects_hidden_and_background_centers(self):
        from sdf import SurfaceSDF, visible_primitives
        # Analytic front plane at camera Z=2; one near-surface primitive.
        means = torch.tensor([[0.,0.,2.02], [0.,0.,2.5], [3.,0.,2.], [0.,0.,-1.],
                              [0.,0.,2.]], requires_grad=True)
        opacity = torch.tensor([.9,.9,.9,.9,.1], requires_grad=True)
        depth = torch.full((9,9,1), 2., requires_grad=True)
        alpha = torch.ones_like(depth)
        sample = dict(viewmat=torch.eye(4), K=torch.tensor([[10.,0.,4.5],[0.,10.,4.5],[0.,0.,1.]]), alpha=alpha)
        selected = visible_primitives(means, opacity, {'surface_depth':depth}, alpha, sample, 1.)
        self.assertEqual(selected.tolist(), [0])
        field = SurfaceSDF(torch.tensor([0.,0.,2.]), 1.)
        loss = field.primitive_loss(means[selected])
        loss.backward()
        self.assertGreater(float(means.grad[0].abs().sum()), 0.)
        self.assertEqual(float(means.grad[1:].abs().sum()), 0.)
        self.assertIsNone(opacity.grad)
        self.assertIsNone(depth.grad)
        self.assertTrue(all(p.grad is None for p in field.parameters()))

    def test_sdf_shading_image_gradient_and_inference_agree(self):
        from sdf import SurfaceSDF
        torch.manual_seed(27)
        field = SurfaceSDF(torch.zeros(3), 2.)
        points = torch.tensor([[.3, .2, 1.], [-.4, .3, .8]], requires_grad=True)
        eye = torch.tensor([0., 0., 4.])
        normal = field.shading_normals(points, eye)
        with torch.no_grad():
            inference = field.shading_normals(points.detach(), eye)
        torch.testing.assert_close(inference, normal.detach(), rtol=0, atol=0)
        light = F.normalize(torch.tensor([.5, -.3, 1.]), dim=0)
        color = (normal*light).sum(-1).clamp_min(0)
        (color-.4).square().mean().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(float(points.grad.abs().sum()), 0.)
        gradients = [p.grad for p in field.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_sdf_mutual_supervision_gradient_ownership(self):
        from sdf import SurfaceSDF
        torch.manual_seed(17)
        field = SurfaceSDF(torch.zeros(3), 1.)
        # Oriented observations independent of the field predictions.
        points = (torch.rand(64, 3)*.4).requires_grad_()
        normals = F.normalize(torch.tensor([.3, .2, -1.]).expand_as(points).clone(), dim=-1).requires_grad_()
        fitting = field.fit_losses(points, normals, normals)
        sum(fitting.values()).backward()
        self.assertIsNone(points.grad)
        self.assertIsNone(normals.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in field.parameters()))
        field.zero_grad(set_to_none=True)
        geometry = field.geometry_losses(points, normals, 1., 1.)
        sum(geometry.values()).backward()
        self.assertTrue(all(p.grad is None for p in field.parameters()))
        for tensor in (points, normals):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.)

    def test_sdf_signed_distance_fit_on_sphere(self):
        from sdf import SurfaceSDF
        torch.manual_seed(11)
        field = SurfaceSDF(torch.zeros(3), 2.)
        normal = F.normalize(torch.randn(128, 3), dim=-1)
        points = normal*.4
        optimizer = torch.optim.Adam(field.parameters(), lr=.002)
        before = field(points).abs().mean().item()
        for _ in range(80):
            optimizer.zero_grad()
            sum(field.fit_losses(points, normal, normal).values()).backward()
            optimizer.step()
        self.assertLess(field(points).abs().mean().item(), before*.15)
        self.assertGreater(float((field(points+.04*normal)>0).float().mean()), .95)
        self.assertGreater(float((field(points-.04*normal)<0).float().mean()), .95)
        restored = SurfaceSDF(torch.ones(3), 3.)
        restored.load_state_dict(field.state_dict())
        torch.testing.assert_close(restored(points), field(points), rtol=0, atol=0)

    def test_camera_depth_normal_orientation_and_gradient(self):
        k = torch.tensor([[12., 0., 4.5], [0., 12., 4.5], [0., 0., 1.]])
        depth = torch.full((9, 9, 1), 2., requires_grad=True)
        normal = depth_normals(depth, k)
        expected = torch.tensor([0., 0., -1.]).expand(7, 7, 3)
        torch.testing.assert_close(normal[1:-1, 1:-1], expected)
        tilted = F.normalize(torch.tensor([.3, 0., -1.]), dim=0)
        loss = (1-(normal[1:-1, 1:-1]*tilted).sum(-1)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertGreater(float(depth.grad.abs().sum()), 0.)

    def test_da3_depth_loss_ignores_scale_but_penalizes_shape(self):
        depth = torch.linspace(1., 3., 81).reshape(9, 9, 1).requires_grad_()
        normal = torch.tensor([0., 0., -1.]).expand(9, 9, 3)
        info = dict(surface_depth=depth, surface_normals=normal, surface_distortion=torch.zeros_like(depth))
        sample = dict(K=torch.eye(3), viewmat=torch.eye(4), alpha=None)
        alpha = torch.ones_like(depth)
        same = surface_losses(info, alpha, sample, {'depth': depth.detach()*7}, 1., 0., 1., 0., 0.)
        self.assertLess(float(same['depth_prior']), 1e-10)
        wrong = surface_losses(info, alpha, sample, {'depth': depth.detach().flip(0)}, 1., 0., 1., 0., 0.)
        self.assertGreater(float(wrong['depth_prior']), .1)
        wrong['depth_prior'].backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertGreater(float(depth.grad.abs().sum()), 0.)

    def test_disk_mass_and_split_keep_tangent_geometry_and_optimizer(self):
        g = Gaussians(3, torch.zeros(3), 1., device='cpu', geometry='2dgs')
        with torch.no_grad():
            g.params['means'].zero_()
            g.params['quats'].zero_(); g.params['quats'][:, 0] = 1
            g.params['scales'].copy_(torch.tensor([[1., 2.], [2., 2.], [3., 2.]]).log())
        torch.testing.assert_close(quadrature_mass(g.params), torch.tensor([1., 2., 3.])/6)
        optimizers = g.optimizers()
        sum(p.square().sum() for p in g.params.values()).backward()
        for optimizer in optimizers.values():
            optimizer.step(); optimizer.zero_grad()
        state = {'count': torch.ones(3), 'scene_scale': 1.}
        split_surfels(g.params, optimizers, state, torch.tensor([True, False, False]))
        self.assertEqual(g.params['scales'].shape, (4, 2))
        self.assertEqual(state['count'].shape, (4,))
        torch.testing.assert_close(g.params['means'][:, 2], torch.zeros(4))
        for key, parameter in g.params.items():
            self.assertIs(optimizers[key].param_groups[0]['params'][0], parameter)
            self.assertEqual(optimizers[key].state[parameter]['exp_avg'].shape, parameter.shape)


class NeuralMaterialGeometryTests(unittest.TestCase):

    @unittest.skipUnless(torch.cuda.is_available(), '3D neural material renderer integration')
    def test_gggs_neural_material_frozen_decoder_geometry_gradients_and_reload(self):
        self.check_gggs_3d_material('neural_material')

    def check_gggs_3d_material(self, representation):
        import tempfile
        from pathlib import Path
        from renderer import render,covariance_normals
        from evaluate import load_model
        config=resolve_config(dict(representation=representation,init_geometry_format='gggs',rank=4))
        self.assertEqual(config['geometry'],'3dgs')
        self.assertEqual(resolve_config(dict(representation=representation))['geometry'],'2dgs')
        g=Gaussians(25,torch.zeros(3,device='cuda'),1.,feature_dim=config['feature_dim'],geometry='3dgs')
        with torch.no_grad():
            xy=torch.cartesian_prod(torch.linspace(-.6,.6,5,device='cuda'),torch.linspace(-.6,.6,5,device='cuda'))
            g.params['means'].copy_(torch.cat((xy,torch.full((25,1),2.,device='cuda')),1))
            g.params['scales'].copy_(torch.tensor([.22,.18,.04],device='cuda').log())
            g.params['opacities'].fill_(0.)
        t=build_transport(config,1.).cuda()
        if representation == 'neural_material':
            # A real neural decoder fixture; scene smoke uses the trained prior.
            with tempfile.TemporaryDirectory() as directory:
                prior=Path(directory)/'prior.pt'
                torch.save(dict(decoder=t.decoder.state_dict(),initial_latent=torch.full((6,),.5)),prior)
                t.material_decoder=str(prior);t.initialize_material(g)
            frozen={k:v.detach().clone() for k,v in t.decoder.state_dict().items()}
            t.requires_grad_(True)
        else:
            t.initialize_material(g)
        s=dict(image=torch.zeros(32,32,3,device='cuda'),K=torch.tensor([[40.,0,16.],[0,40.,16.],[0,0,1.]],device='cuda'),
               viewmat=torch.eye(4,device='cuda'),c2w=torch.eye(4,device='cuda'),
               light_pos=torch.tensor([.5,-.3,0.],device='cuda'),light_intensity=torch.ones(3,device='cuda'))
        torch.testing.assert_close(covariance_normals(g,s['c2w'][:3,3]),
            torch.tensor([0.,0.,-1.],device='cuda').expand(25,3))
        rgb,alpha,info=render(g,t,s,background=0,shadow=False)
        self.assertTrue(torch.isfinite(rgb).all())
        (rgb.square().mean()+alpha.mean()).backward()
        for name in ['means','scales','quats','opacities','features','base']:
            grad=g.params[name].grad
            self.assertIsNotNone(grad,name);self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(float(grad.abs().sum()),0.,name)
        if representation == 'neural_material':
            for name,value in t.decoder.named_parameters():
                self.assertFalse(value.requires_grad,name)
                self.assertIsNone(value.grad,name)
                torch.testing.assert_close(value,frozen[name],rtol=0,atol=0)
            self.assertGreater(float(g.params['features'].grad[:,:6].abs().sum()),0.)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'last.pt'
            torch.save(dict(config=config,radius=g.radius,gaussians=g.state_dict(),transport=t.state_dict()),p)
            restored,material,_=load_model(p)
            self.assertEqual(restored.params['scales'].shape[-1],3)
            actual,_,_=render(restored,material,s,background=0,shadow=False)
            torch.testing.assert_close(actual,rgb,atol=1e-6,rtol=1e-5)


class BoundaryGeometryTests(unittest.TestCase):
    def test_spike_and_shrink_are_not_both_called_improvements(self):
        from native_reconstruction import boundary_metrics
        gt=torch.zeros(32,32);gt[8:24,8:24]=1
        normal=torch.zeros(32,32,3);normal[...,2]=-1
        def measure(mask):return boundary_metrics(dict(alpha=mask,depth=mask*3,surface_normal=normal),dict(alpha=gt[...,None]))
        correct=measure(gt);self.assertEqual(correct['boundary_F1_2px'],1.)
        spike=gt.clone();spike[15:17,24:31]=1
        self.assertGreater(measure(spike)['boundary_outward_fraction_over_2px'],0.)
        shrunk=torch.zeros_like(gt);shrunk[12:20,12:20]=1
        bad=measure(shrunk)
        self.assertEqual(bad['boundary_outward_fraction_over_2px'],0.)
        self.assertLess(bad['boundary_recall_2px'],correct['boundary_recall_2px'])


class GroundedGeometryTests(unittest.TestCase):

    def test_centered_canvas_preserves_off_center_pixel_rays(self):
        from gggs_reconstruction import centered_canvas
        from multiview_geometry import pixel_rays
        K=torch.tensor([[90.,0,26.7],[0,95.,15.3],[0,0,1.]])
        sample=dict(K=K,image=torch.zeros(48,64,3),viewmat=torch.eye(4),c2w=torch.eye(4))
        camera,grid=centered_canvas(sample)
        uv=(grid+1)*torch.tensor([camera.image_width,camera.image_height])/2
        rays=torch.cat(((uv-torch.tensor([camera.image_width/2,camera.image_height/2]))/K.diag()[:2],torch.ones(48,64,1)),-1)
        torch.testing.assert_close(rays,pixel_rays(sample),atol=1e-6,rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA continuous-depth rasterizer')
    def test_continuous_plane_depth_gradients_and_world_scale(self):
        import numpy as np
        from types import SimpleNamespace
        from gggs_reconstruction import author_modules,render_geometry
        from native_reconstruction import reconstruction_coordinates
        Model,opt,render=author_modules()
        xy=torch.cartesian_prod(torch.linspace(-1,1,30),torch.linspace(-1,1,30))
        xyz=torch.cat((xy,torch.full((len(xy),1),3.)),1).numpy()
        model=Model(3,0);model.create_from_pcd(SimpleNamespace(_xyz=xyz,_rgb=np.full_like(xyz,.5)),1.)
        model.create_app_model(1,Model.App_model.NO);model.training_setup(opt);model.reset_3D_filter()
        with torch.no_grad():
            model._scaling[:,:2]=math.log(.045);model._scaling[:,2]=math.log(.004)
            model._opacity.fill_(4.)
        sample=dict(K=torch.tensor([[90.,0,26.7],[0,95.,15.3],[0,0,1.]],device='cuda'),
                    image=torch.zeros(48,64,3,device='cuda'),viewmat=torch.eye(4,device='cuda'),c2w=torch.eye(4,device='cuda'))
        result=render_geometry(model,sample,render)
        self.assertGreater(float(result['alpha'][24,32]),.95)
        self.assertLess(abs(float(result['depth'][24,32])-3.),.03)
        self.assertLess(float(result['normal'][24,32,2]),-.95)
        objective=result['depth'][16:32,16:40].mean()+result['rgb'].mean()+result['alpha'].mean()
        objective.backward()
        self.assertTrue(bool(torch.isfinite(model._xyz.grad).all()))
        self.assertGreater(float(model._xyz.grad.norm()),0.)
        center=torch.tensor([.2,-.4,.5],device='cuda');scale=4.
        with torch.no_grad():
            model._xyz.copy_((model._xyz-center)/scale);model._scaling.sub_(math.log(scale))
            transformed=render_geometry(model,reconstruction_coordinates(sample,center,scale),render)
        torch.testing.assert_close(transformed['rgb'],result['rgb'],atol=3e-4,rtol=1e-3)
        torch.testing.assert_close(transformed['depth'][24,32]*scale,result['depth'][24,32],atol=3e-4,rtol=1e-3)

        # Mip footprint must survive normalized-training -> world export/reload.
        from gggs_reconstruction import centered_canvas,restore_model
        local=reconstruction_coordinates(sample,center,scale)
        model.compute_3D_filter([centered_canvas(local)[0]])
        self.assertTrue(bool((model.filter_3D>0).all()))
        filtered=render_geometry(model,local,render)
        payload=dict(config=dict(optimizer=vars(opt),mip_filter=True,background=0.,
                                 training_coordinate_transform=dict(scale=scale,center=center.tolist())),capture=model.capture(),
                     world_filter_3D=model.filter_3D*scale,
                     gaussians={'params.means':model._xyz*scale+center,'params.scales':model._scaling+math.log(scale),
                                'params.quats':model._rotation,'params.opacities':model._opacity[:,0]})
        stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
        restored=restore_model(Model,torch.load(stream,weights_only=False))
        replay=render_geometry(restored,sample,render)
        torch.testing.assert_close(replay['rgb'],filtered['rgb'],atol=3e-4,rtol=1e-3)
        torch.testing.assert_close(replay['depth'][24,32],filtered['depth'][24,32]*scale,atol=3e-4,rtol=1e-3)
        # Evaluation should retain training coordinates inside the depth solver.
        from gggs_reconstruction import restore_evaluation_model,render_evaluation_geometry
        stream.seek(0);saved=torch.load(stream,weights_only=False)
        exact=render_evaluation_geometry(restore_evaluation_model(Model,saved),sample,render,saved['config'])
        torch.testing.assert_close(exact['depth'],filtered['depth']*scale,atol=1e-6,rtol=1e-6)


class GeometryImportTests(unittest.TestCase):
    def test_gggs_default_handoff_preserves_filtered_world_geometry(self):
        from gggs_reconstruction import relighting_state
        raw=torch.tensor([[.1,.2,.3],[.4,.2,.1]])
        means=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        center=torch.tensor([.5,-.3,.2]);scale=2.
        capture=[None]*8
        capture[2]=means;capture[5]=raw.log()
        capture[6]=torch.tensor([[2.,0,0,0],[0.,3.,0,0]])
        capture[7]=torch.tensor([[.2],[-.3]])
        source=dict(kind='gggs_core',config={'training_coordinate_transform':{'scale':scale}},
                    capture=capture,world_filter_3D=torch.tensor([[.04],[.06]]),
                    gaussians={'center':center},radius=3.)
        result=relighting_state(source)
        expected=torch.sqrt((raw*scale).square()+source['world_filter_3D'].square())
        opacity=capture[7][:,0].sigmoid()*((raw*scale).prod(-1)/expected.prod(-1))
        torch.testing.assert_close(result['params.means'],means*scale+center)
        torch.testing.assert_close(result['params.scales'].exp(),expected)
        torch.testing.assert_close(result['params.opacities'].sigmoid(),opacity)
        torch.testing.assert_close(result['params.quats'].norm(dim=-1),torch.ones(2))
        self.assertTrue(all(not x.requires_grad for x in result.values()))
        with self.assertRaises(ValueError):relighting_state(dict(source,kind='native_2dgs'))


class MultiviewGeometryTests(unittest.TestCase):
    def test_full_intrinsics_and_rotated_camera_roundtrip(self):
        from multiview_geometry import world_points, project_world
        angle=torch.tensor(.31)
        R=torch.tensor([[angle.cos(),0,angle.sin()],[0,1,0],[-angle.sin(),0,angle.cos()]])
        view=torch.eye(4);view[:3,:3]=R;view[:3,3]=torch.tensor([.2,-.1,.5])
        sample=dict(image=torch.zeros(24,32,3),K=torch.tensor([[55.,0,13.2],[0,61.,8.7],[0,0,1.]]),viewmat=view)
        depth=torch.linspace(2,4,24*32).reshape(24,32)
        grid,z=project_world(world_points(sample,depth),sample)
        y,x=torch.meshgrid(torch.arange(24)+.5,torch.arange(32)+.5,indexing='ij')
        expected=torch.stack((x/32*2-1,y/24*2-1),-1)
        torch.testing.assert_close(grid,expected,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(z,depth)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA plane-sweep regression')
    def test_known_textured_plane_with_exposure_change_and_blank_rejection(self):
        from multiview_geometry import pixel_rays, plane_sweep
        def camera(x,gain):
            K=torch.tensor([[90.,0,29.7],[0,95.,22.3],[0,0,1.]],device='cuda')
            view=torch.eye(4,device='cuda');view[0,3]=-x
            s=dict(K=K,viewmat=view,image=torch.zeros(48,64,3,device='cuda'),alpha=torch.ones(48,64,1,device='cuda'))
            xyz=pixel_rays(s)*4;xyz[...,0]+=x
            value=.45+.15*torch.sin(xyz[...,0]*19+xyz[...,1]*7)+.12*torch.cos(xyz[...,1]*23-xyz[...,0]*9)
            s['image']=(gain*value+.03)[...,None].expand(-1,-1,3)
            return s
        reference=camera(0,1)
        sources=[camera(x,g) for x,g in [(-.3,.7),(-.15,1.1),(.15,.8),(.3,1.2)]]
        center=torch.tensor([0.,0,4.],device='cuda')
        result=plane_sweep(reference,sources,center,.5,128)
        valid=result['reliable'][8:-8,10:-10]
        self.assertGreater(float(valid.float().mean()),.2)
        error=(result['depth'][8:-8,10:-10][valid]-4).abs().median()
        self.assertLess(float(error),.02)
        for sample in [reference,*sources]:sample['image'].fill_(.5)
        blank=plane_sweep(reference,sources,center,.5,32)
        self.assertFalse(bool(blank['reliable'].any()))


class CameraCalibrationTests(unittest.TestCase):
    """Fit-camera rotation corrections and the secondary shift-aligned metric."""

    @staticmethod
    def _sample(angle=0.3):
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.linalg.matrix_exp(torch.tensor([[0., -angle, 0.], [angle, 0., 0.], [0., 0., 0.]]))
        c2w[:3, 3] = torch.tensor([1.5, -0.4, 4.0])
        flip = torch.diag(torch.tensor((1., -1., -1., 1.)))
        return {'c2w': c2w, 'viewmat': torch.linalg.inv(c2w @ flip), 'image': torch.zeros(4, 4, 3)}

    def test_rotation_mode_preserves_camera_centers_and_starts_at_identity(self):
        from cameras import TrainCameraRotations, build_camera_offsets, load_camera_offsets
        offsets = build_camera_offsets('rotation', 3, 1.0)
        self.assertIsInstance(offsets, TrainCameraRotations)
        sample = self._sample()
        torch.testing.assert_close(offsets.correct(sample, 1)['viewmat'], sample['viewmat'])
        with torch.no_grad():
            offsets.rotation.weight[1] = torch.tensor([0.01, -0.02, 0.005])
        corrected = offsets.correct(sample, 1)
        torch.testing.assert_close(corrected['c2w'][:3, 3], sample['c2w'][:3, 3], rtol=0, atol=1e-5)
        self.assertGreater((corrected['viewmat'] - sample['viewmat']).abs().max().item(), 1e-3)
        torch.testing.assert_close(offsets.correct(sample, 0)['viewmat'], sample['viewmat'])
        state = {'camera_offsets': offsets.state_dict(), 'fit_indices': [4, 7, 9],
                 'config': {'camera_mode': 'rotation'}}
        restored = load_camera_offsets(state, 1.0)
        torch.testing.assert_close(restored.correct(sample, 1)['viewmat'], corrected['viewmat'])
        legacy = load_camera_offsets({**state, 'config': {}, 'camera_offsets': build_camera_offsets('anchor', 3, 1.0).state_dict()}, 1.0)
        self.assertEqual(type(legacy).__name__, 'TrainCameraOffsets')

    def test_rotation_mode_updates_only_the_sampled_camera(self):
        from cameras import build_camera_offsets
        offsets = build_camera_offsets('rotation', 4, 1.0)
        optimizer = offsets.optimizer(1e-2)
        sample = self._sample()
        target = torch.tensor([0.3, 0.2, 5.0])
        corrected = offsets.correct(sample, 2)
        point = corrected['viewmat'][:3, :3] @ target + corrected['viewmat'][:3, 3]
        loss = point[:2].sum() + offsets.regularization(2)
        loss.backward()
        self.assertTrue(offsets.rotation.weight.grad.is_sparse)
        optimizer.step()
        moved = offsets.rotation.weight.detach().abs().sum(-1) > 0
        self.assertEqual(moved.tolist(), [False, False, True, False])

    def test_translation_gauge_removal_preserves_center_images_and_is_idempotent(self):
        from cameras import build_camera_offsets
        torch.manual_seed(3)
        views, center = [], torch.tensor([0.1, -0.2, 0.3])
        for angle in torch.linspace(0, 2 * math.pi, 9)[:-1]:
            eye = center + torch.stack((5 * angle.cos(), torch.tensor(2.0), 5 * angle.sin()))
            forward = F.normalize(center - eye, dim=0)
            right = F.normalize(torch.linalg.cross(forward, torch.tensor([0., 1., 0.])), dim=0)
            down = torch.linalg.cross(forward, right)
            rotation = torch.stack((right, down, forward))
            view = torch.eye(4)
            view[:3, :3], view[:3, 3] = rotation, -rotation @ eye
            views.append(view)
        views = torch.stack(views)
        intrinsics = torch.tensor([[3000., 0., 240.], [0., 3000., 250.], [0., 0., 1.]]).expand(len(views), 3, 3)
        offsets = build_camera_offsets('rotation', len(views), 1.0)
        offsets.set_translation_gauge(views, intrinsics, center)
        with torch.no_grad():
            offsets.rotation.weight.copy_(1e-3 * torch.randn(len(views), 3))

        def center_pixels(scene_shift):
            pixels = []
            for index, view in enumerate(views):
                corrected = offsets.correct({'viewmat': view}, index)['viewmat']
                point = corrected[:3, :3] @ (center + scene_shift) + corrected[:3, 3]
                pixels.append((intrinsics[index] @ point)[:2] / point[2])
            return torch.stack(pixels)
        before = center_pixels(torch.zeros(3))
        shift = offsets.remove_translation_gauge()
        after = center_pixels(shift)
        self.assertLess((after - before).abs().max().item(), 0.02)
        self.assertGreater(shift.norm().item(), 1e-5)
        balance = (offsets.gauge.transpose(1, 2) @ offsets.rotation.weight[..., None]).sum(0)
        self.assertLess(balance.abs().max().item(), 1e-7)
        self.assertLess(offsets.remove_translation_gauge().norm().item(), 1e-7)

    def test_light_offsets_move_only_fit_light_rows(self):
        from cameras import TrainLightOffsets
        lights = TrainLightOffsets(3, 2.0)
        optimizer = lights.optimizer(1e-2)
        sample = {'light_pos': torch.tensor([1., 2., 3.])}
        torch.testing.assert_close(lights.correct(sample, 1)['light_pos'], sample['light_pos'])
        corrected = lights.correct(sample, 1)
        (corrected['light_pos'].sum() + lights.regularization(1)).backward()
        optimizer.step()
        moved = lights.offset.weight.detach().abs().sum(-1) > 0
        self.assertEqual(moved.tolist(), [False, True, False])
        self.assertAlmostEqual(lights.correct(sample, 1)['light_pos'][0].item() - 1.0,
                               2.0 * lights.offset.weight[1, 0].item(), places=6)

    def test_shift_alignment_recovers_subpixel_translation(self):
        from evaluate import shift_images, shift_aligned_metrics
        y, x = torch.meshgrid(torch.arange(48.), torch.arange(48.), indexing='ij')
        target = torch.stack((torch.exp(-((x-20)**2+(y-26)**2)/40), torch.exp(-((x-28)**2+(y-18)**2)/25),
                              0.5*torch.exp(-((x-24)**2+(y-24)**2)/90)), -1)
        moved = shift_images(target, torch.tensor([[2.25, -3.5]]))[0]
        item = shift_aligned_metrics(moved, target, radius=6)
        self.assertAlmostEqual(item['dy'], -2.25, delta=0.13)
        self.assertAlmostEqual(item['dx'], 3.5, delta=0.13)
        self.assertGreater(item['PSNR'], 35.0)


class TrainingStructureTests(unittest.TestCase):
    """Training package contracts: CLI checks, frozen stages, refinement and partition memory."""

    @staticmethod
    def parse(*extra):
        import contextlib
        from training.options import parse_arguments
        with contextlib.redirect_stderr(io.StringIO()):
            return parse_arguments(['--scene', 'scene', '--output', 'out', *extra])

    def test_cli_rejects_zero_validation_limit_and_negative_reset_interval(self):
        for extra in (['--val-limit', '0'], ['--opacity-reset-every', '-1']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.parse(*extra)
        self.assertEqual(self.parse('--opacity-reset-every', '0').opacity_reset_every, 0)

    def test_residual_stage_freezes_geometry_and_camera_rate(self):
        args = self.parse('--representation', 'neural_material', '--radiance-residual', '--init-checkpoint', 'x.pt')
        self.assertTrue(args.freeze_geometry)
        self.assertEqual((args.camera_lr, args.camera_start), (0., 1))

    def test_frozen_stage_reuses_rotation_corrections_without_an_optimizer(self):
        from types import SimpleNamespace
        from cameras import build_camera_offsets
        from training.pose import CameraFit
        # SparseAdam rejects the frozen stage's zero rate, which crashed residual stages.
        with self.assertRaises(ValueError):
            build_camera_offsets('rotation', 3, 1.).optimizer(0.)
        source = build_camera_offsets('rotation', 3, 1.)
        with torch.no_grad():
            source.rotation.weight[1] = torch.tensor([.01, -.02, .005])
        saved = {'camera_offsets': source.state_dict(), 'config': {'camera_mode': 'rotation'}, 'fit_indices': [0, 2, 5]}
        args = SimpleNamespace(init_checkpoint='x.pt', sdf_volume_only=False, radiance_residual=True,
                               camera_mode='anchor', camera_lr=0., camera_lr_final=None, camera_start=1,
                               steps=4, camera_gauge='none')
        config = {'camera_mode': 'anchor'}
        gaussians = SimpleNamespace(radius=1., center=torch.zeros(3))
        camera = CameraFit(args, config, saved, None, [0, 2, 5], gaussians, trainable=False)
        self.assertFalse(camera.trainable)
        self.assertEqual(config['camera_mode'], 'rotation')
        self.assertFalse(any(parameter.requires_grad for parameter in camera.offsets.parameters()))
        viewmat = torch.eye(4)
        sample = {'viewmat': viewmat, 'c2w': viewmat.clone()}
        corrected = camera.begin(1, sample, 2)
        torch.testing.assert_close(corrected['viewmat'], source.correct(sample, 1)['viewmat'], rtol=0, atol=0)
        self.assertEqual(camera.log_fields()['camera_lr'], 0.)

    def test_opacity_reset_interval_is_separate_from_pruning_schedule(self):
        from refinement import Refinement
        events = {}
        for interval in (None, 0):
            params = torch.nn.ParameterDict({'means': torch.nn.Parameter(torch.zeros(4, 3)),
                                             'opacities': torch.nn.Parameter(torch.full((4,), 3.))})
            optimizers = {'opacities': torch.optim.Adam([params['opacities']])}
            strategy = Refinement(refine_start_iter=10**9, reset_every=3000, opacity_reset_every=interval)
            strategy._update_state = lambda *args, **kwargs: None
            events[interval] = strategy.step_post_backward(params, optimizers, {}, 3000, {})
            self.assertEqual(strategy.reset_every, 3000)
            if interval == 0:
                torch.testing.assert_close(params['opacities'].detach(), torch.full((4,), 3.), rtol=0, atol=0)
        self.assertEqual(events[None]['event'], 'opacity_reset')
        self.assertIsNone(events[0])

    def test_chunked_partition_matches_direct_values_and_gradients(self):
        import methods.base as base
        torch.manual_seed(3)
        inputs = (torch.randn(29, 3), torch.rand(6, 3) - .5, torch.full((6,), .3).log())
        results = []
        for function in (base._log_partition, base.spatial_partition):
            leaves = [value.clone().requires_grad_() for value in inputs]
            rows, base._PARTITION_ROWS = base._PARTITION_ROWS, 7
            try:
                output = function(*leaves)
            finally:
                base._PARTITION_ROWS = rows
            (output.exp() * torch.arange(6.)).sum().backward()
            results.append((output.detach(), [leaf.grad for leaf in leaves]))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        for direct, chunked in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(direct, chunked, rtol=1e-6, atol=1e-7)

    def test_manifest_records_source_provenance(self):
        from pathlib import Path
        from make_validation_manifest import build_manifest
        config = dict(name='provenance_probe', python='python', data_root='/data', worker_gpus=[0],
                      train={}, families={'family': {'scenes': ['scene'], 'train': {}}}, launch_environment={})
        manifest = build_manifest(config, Path('/tmp/port-provenance-probe'))
        provenance = manifest['source_provenance']
        self.assertEqual(provenance['revision'], manifest['source_revision'])
        self.assertEqual(provenance['dirty'], bool(provenance['status']))
        self.assertEqual(manifest['jobs'][0]['source_dirty'], provenance['dirty'])
        self.assertEqual(len(provenance['diff_sha256']), 64)


class LightAtlasTests(unittest.TestCase):
    """Light-space atlas conventions on a floor with an occluding square under an overhead light."""

    @staticmethod
    def scene():
        axis = torch.linspace(-1, 1, 48)
        floor = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
        floor = torch.stack((floor[:, 0], torch.zeros(len(floor)), floor[:, 1]), -1)
        axis = torch.linspace(-.25, .25, 12)
        square = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
        square = torch.stack((square[:, 0], torch.full((len(square),), .5), square[:, 1]), -1)
        means = torch.cat((floor, square)).cuda()
        g = Gaussians(len(means), torch.tensor([0., .25, 0.]), 1.5, 8, device='cuda')
        with torch.no_grad():
            g.params['means'].copy_(means)
            g.params['quats'].copy_(torch.tensor([1., 0, 0, 0]).expand(len(means), 4))
            scales = torch.tensor([.045, .002, .045]).expand(len(means), 3).clone()
            scales[len(floor):] = torch.tensor([.03, .002, .03])
            g.params['scales'].copy_(scales.log())
            g.params['opacities'].fill_(4.)
        return g

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA light-space rasterization')
    def test_moment_visibility_matches_cast_shadow_and_reaches_occluder(self):
        torch.manual_seed(0)
        g = self.scene()
        transport = build_transport(dict(representation='light_atlas', feature_dim=8), 1.).cuda()
        light = torch.tensor([0., 3., 0.], device='cuda')
        # Shadow of the |x|,|z|<=.25 square at height .5 spans |x|<=.3 on the floor.
        receivers = torch.tensor([[0., 0., 0.], [.15, 0., -.1], [.8, 0., .8], [.5, 0., 0.], [0., .5, 0.]],
                                 device='cuda')
        atlas = transport.light_atlas(g, light)
        self.assertEqual([level.shape[-1] for level in atlas['levels']], [512, 256, 128, 64, 32, 16, 8])
        stats, fluxes = transport.gather(atlas, receivers, g.radius)
        self.assertEqual(stats.shape, (5, 5 * 7))
        self.assertEqual(fluxes.shape, (5, 7, 8))
        visibility = stats[:, 4]
        self.assertTrue((visibility[:2] < .05).all(), visibility)
        self.assertTrue((visibility[2:] > .95).all(), visibility)
        occluder = torch.autograd.grad(stats[0, 4], g.params['means'])[0][48 * 48:]
        self.assertTrue(torch.isfinite(occluder).all() and occluder.abs().sum() > 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA light-space rasterization')
    def test_inactive_light_pass_is_local_shading_and_transport_is_flux_linear(self):
        torch.manual_seed(0)
        g = self.scene()
        transport = build_transport(dict(representation='light_atlas', feature_dim=8), 1.).cuda()
        points = torch.tensor([[.6, 0., .6], [0., .5, 0.]], device='cuda')
        receivers = {'means': points, 'features': torch.randn(2, 8, device='cuda'),
                     'base': torch.zeros(2, 3, device='cuda'),
                     'normals': torch.tensor([[0., 1., 0.]], device='cuda').expand(2, 3)}
        eye, light = torch.tensor([0., 2., 3.], device='cuda'), torch.tensor([0., 3., 0.], device='cuda')
        intensity = torch.ones(3, device='cuda')
        local = transport(g, receivers, eye, light, intensity, None, port_active=False, shadow=False)
        delta = light - points
        rho, _ = transport.local_response(receivers, receivers['normals'], F.normalize(delta, dim=-1),
                                          F.normalize(eye - points, dim=-1),
                                          F.normalize(F.normalize(delta, dim=-1) + F.normalize(eye - points, dim=-1), dim=-1))
        torch.testing.assert_close(local, rho / delta.square().sum(-1, keepdim=True))
        stats, fluxes = transport.gather(transport.light_atlas(g, light), points, g.radius)
        cosines = torch.randn(2, 5, device='cuda')
        torch.testing.assert_close(transport.transfer(stats, 2 * fluxes, receivers['features'], cosines),
                                   2 * transport.transfer(stats, fluxes, receivers['features'], cosines))
        full = transport(g, receivers, eye, light, intensity, None, port_active=True, shadow=True)
        self.assertTrue(torch.isfinite(full).all() and (full >= 0).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA light-space rasterization')
    def test_material_heads_and_per_gaussian_visibility_ablation(self):
        torch.manual_seed(0)
        g = self.scene()
        points = torch.tensor([[0., 0., 0.], [.8, 0., .8]], device='cuda')
        receivers = {'means': points, 'features': torch.randn(2, 8, device='cuda'),
                     'base': torch.zeros(2, 3, device='cuda'),
                     'normals': torch.tensor([[0., 1., 0.]], device='cuda').expand(2, 3),
                     'visibility': torch.tensor([.25, 1.], device='cuda')}
        eye, light = torch.tensor([0., 2., 3.], device='cuda'), torch.tensor([0., 3., 0.], device='cuda')
        intensity = torch.ones(3, device='cuda')
        compact = build_transport(dict(representation='light_atlas', feature_dim=8), 1.)
        spatial = build_transport(dict(representation='light_atlas', feature_dim=8, material_head='spatial'), 1.)
        # The compact head keeps the first-run parameter layout so its checkpoints still load.
        self.assertEqual(compact.material[0].in_features, 8 + 75)
        self.assertEqual(len(compact.material), 7)
        self.assertEqual(spatial.material[0].in_features, 8 + 3 + 4 * 27 + 5 + 5 + 51)
        self.assertEqual(len(spatial.material), 9)
        out = spatial.cuda()(g, receivers, eye, light, intensity, None, port_active=True, shadow=True)
        self.assertTrue(torch.isfinite(out).all() and (out >= 0).all())
        ablation = build_transport(dict(representation='light_atlas', feature_dim=8, visibility_model='gaussian',
                                        light_transport='none'), 1.).cuda()
        self.assertTrue(ablation.per_gaussian_visibility)
        shaded = ablation(g, receivers, eye, light, intensity, None, port_active=True, shadow=True)
        unshadowed = ablation(g, receivers, eye, light, intensity, None, port_active=True, shadow=False)
        torch.testing.assert_close(shaded, unshadowed * receivers['visibility'][:, None])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA light-space rasterization')
    def test_svbrdf_head_limits_position_to_light_independent_coefficients(self):
        torch.manual_seed(0)
        g = self.scene()
        head = build_transport(dict(representation='light_atlas', feature_dim=8, material_head='svbrdf'), 1.).cuda()
        self.assertEqual(head.material[0].in_features, 8 + 3 + 4 * 27 + 5 + 5)  # no positional input
        self.assertEqual(head.material[-1].out_features, 3 * 9)
        self.assertEqual(head.coefficients[0].in_features, 8 + 51)
        points = torch.tensor([[0., 0., 0.], [.8, 0., .8]], device='cuda')
        receivers = {'means': points, 'features': torch.randn(2, 8, device='cuda'),
                     'base': torch.zeros(2, 3, device='cuda'),
                     'normals': torch.tensor([[0., 1., 0.]], device='cuda').expand(2, 3)}
        eye, intensity = torch.tensor([0., 2., 3.], device='cuda'), torch.ones(3, device='cuda')
        light = torch.tensor([0., 3., 0.], device='cuda')
        out = head(g, receivers, eye, light, intensity, None, port_active=True, shadow=True)
        self.assertTrue(torch.isfinite(out).all() and (out >= 0).all())
        # Zero-initialized coefficients: the positional branch starts silent.
        xyz = (points - g.center) / g.radius
        with torch.no_grad():
            head.coefficients[-1].bias.fill_(1.)
        first = head.coefficients(torch.cat((receivers['features'], direction_encoding(xyz, 8)), -1))
        torch.testing.assert_close(first, torch.ones_like(first))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA light-space rasterization')
    def test_specular_lobes_use_geometric_shadow_and_bounded_visibility(self):
        torch.manual_seed(0)
        g = self.scene()
        default = build_transport(dict(representation='light_atlas', feature_dim=8, material_head='svbrdf'), 1.)
        # Defaults add no parameters, so checkpoints from before 2026-10-04 keep loading.
        self.assertFalse(any(key.startswith('specular_weights') for key in default.state_dict()))
        self.assertEqual(default.visibility_bound, 0.)
        lobes = build_transport(dict(representation='light_atlas', feature_dim=8, material_head='svbrdf',
                                     specular='lobes', light_transport='none'), 1.).cuda()
        self.assertEqual(lobes.specular_weights[0].in_features, 8 + 51)
        self.assertEqual(lobes.specular_weights[-1].out_features, 3 * 5)
        # Floor point in the cast shadow, lit floor point; mirror geometry for the lit one.
        points = torch.tensor([[0., 0., 0.], [.6, 0., 0.]], device='cuda')
        receivers = {'means': points, 'features': torch.randn(2, 8, device='cuda'),
                     'base': torch.zeros(2, 3, device='cuda'),
                     'normals': torch.tensor([[0., 1., 0.]], device='cuda').expand(2, 3)}
        light, eye = torch.tensor([0., 3., 0.], device='cuda'), torch.tensor([1.2, 3., 0.], device='cuda')
        intensity = torch.ones(3, device='cuda')
        with torch.no_grad():
            lobes.material[-1].weight.zero_()
            lobes.material[-1].bias.fill_(-100.)  # local rho ~ 0: only the lobes remain
            lobes.specular_weights[-1].bias.fill_(1.)
            before = lobes(g, receivers, eye, light, intensity, None, port_active=True, shadow=True)
            lobes.visibility[-1].bias.fill_(-100.)  # learned visibility collapses to 0
            after = lobes(g, receivers, eye, light, intensity, None, port_active=True, shadow=True)
        self.assertTrue((before[0] < 1e-3).all(), before)  # shadowed by the moment test
        self.assertTrue((before[1] > 1e-2).all(), before)
        torch.testing.assert_close(after, before)
        bounded = build_transport(dict(representation='light_atlas', feature_dim=8, visibility_bound=2.), 1.).cuda()
        with torch.no_grad():
            bounded.visibility[-1].bias.fill_(-100.)
            stats, _ = bounded.gather(bounded.light_atlas(g, light), points, g.radius)
            visibility = bounded.receiver_visibility(stats, receivers['features'], torch.ones(2, 1, device='cuda'))
            prior = stats[:, 4:5].clamp(1e-4, 1 - 1e-4)
        torch.testing.assert_close(visibility, torch.sigmoid(torch.logit(prior) - 2 * math.tanh(50.)))


class RefinementBudgetTests(unittest.TestCase):
    def test_point_cap_ramps_linearly_and_defaults_to_fixed_cap(self):
        from refinement import Refinement
        fixed = Refinement(max_points=1000)
        self.assertEqual([fixed.point_cap(step) for step in (0, 500, 10**6)], [1000, 1000, 1000])
        ramp = Refinement(refine_start_iter=500, max_points=1000, budget_ramp=1500, initial_points=100)
        self.assertEqual([ramp.point_cap(step) for step in (0, 500, 1000, 1500, 9000)], [100, 100, 550, 1000, 1000])


class AppearanceWeightTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA rasterization')
    def test_zero_weight_pixels_supervise_coverage_only(self):
        from renderer import render
        torch.manual_seed(0)
        g = LightAtlasTests.scene()
        transport = build_transport(dict(representation='light_atlas', feature_dim=8), 1.).cuda()
        c2w = torch.eye(4, device='cuda')
        c2w[:3, :3] = torch.tensor([[1., 0, 0], [0, 0, 1], [0, -1, 0]], device='cuda')  # look down -y
        c2w[:3, 3] = torch.tensor([0., 3., 0.], device='cuda')
        viewmat = torch.linalg.inv(c2w @ torch.diag(torch.tensor([1., -1., -1., 1.], device='cuda')))
        K = torch.tensor([[40., 0, 32], [0, 40., 32], [0, 0, 1]], device='cuda')
        sample = {'image': torch.zeros(64, 64, 3, device='cuda'), 'viewmat': viewmat, 'K': K, 'c2w': c2w,
                  'light_pos': torch.tensor([0., 3., 1.], device='cuda'), 'light_intensity': torch.ones(3, device='cuda')}
        grads = {}
        for weight in (0., 1.):
            for parameter in [*transport.parameters(), *g.params.values()]:
                parameter.grad = None
            image, alpha, _ = render(g, transport, sample, background=1., shadow=True, port_active=True,
                                     appearance_weight=torch.full((64, 64, 1), weight, device='cuda'))
            self.assertGreater(alpha.mean().item(), .1)
            image.sum().backward()
            grads[weight] = (sum(p.grad.abs().sum() for p in transport.parameters() if p.grad is not None),
                             g.params['opacities'].grad.abs().sum())
        self.assertEqual(float(grads[0.][0]), 0.)
        self.assertGreater(float(grads[1.][0]), 0.)
        self.assertGreater(float(grads[0.][1]), 0.)  # coverage is still supervised


if __name__ == '__main__':
    unittest.main()

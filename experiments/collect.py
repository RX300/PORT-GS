"""Audit supplementary results and publish the compact experiment tables."""

import argparse
import csv
import datetime
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from run_benchmark import job_status_record


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('PSNR', 'SSIM', 'LPIPS')
DESTINATION = ROOT / 'docs/experiments/supplementary'


def metric_values(path, dataset):
    report = json.loads(Path(path).read_text())
    frames = json.loads((Path(dataset) / 'transforms_test.json').read_text())['frames']
    views = report['views']
    assert [row['frame_index'] for row in views] == list(range(len(frames))), path
    for row, frame in zip(views, frames):
        filename = frame['file_path'] if 'file_path' in frame else frame['file_paths'][0]
        assert row['name'] == Path(filename).stem, path
        assert all(math.isfinite(row[key]) for key in METRICS), path
    values = {key: statistics.mean(row[key] for row in views) for key in METRICS}
    assert all(abs(values[key] - report['metrics'][key]) < 1e-5 for key in METRICS), path
    return len(views), values


def runtime_seconds(state):
    phases = [phase for name, phase in state['steps'].items()
              if name in ('train', 'appearance') and phase['state'] == 'completed']
    return sum((datetime.datetime.fromisoformat(phase['finished_utc']) -
                datetime.datetime.fromisoformat(phase['started_utc'])).total_seconds()
               for phase in phases)


def reference_rows(manifest, states):
    previous = ROOT / 'runs/lisa_staged_full_dataset_20261004'
    source = json.loads((ROOT.parent / 'OLAT-Gaussians/runs/full_dataset/comparison_manifest.json').read_text())
    references = []
    for item in source['jobs']:
        if item['method'] in ('LiSA-staged', 'GS3', 'SSD-GS', 'RNG', 'OLAT-Gaussians'):
            references.append(dict(item, group='baseline_comparison', variant=item['method'], seed=0,
                                   source_state='completed', reference=True))
    families = manifest['protocol']['families']
    for family, settings in families.items():
        for scene in settings['scenes']:
            output = previous / family / scene / 'appearance'
            item = dict(method='LiSA-staged', family=family, scene=scene, variant='full', seed=0,
                        dataset=str(Path(manifest['data_root']) / family / scene),
                        resultpath=str(output / 'test/metrics.json'), output=str(output),
                        source_state='completed', reference=True)
            references += [dict(item, group='structure_ablation'), dict(item, group='repeatability')]
    for family, settings in manifest['protocol']['groups']['refinement']['families'].items():
        for scene in settings['scenes']:
            output = previous / family / scene / 'appearance'
            references.append(dict(method='LiSA-staged', group='refinement', family=family, scene=scene,
                                   variant='pixel_observation', seed=0,
                                   dataset=str(Path(manifest['data_root']) / family / scene),
                                   resultpath=str(output / 'test/metrics.json'), output=str(output),
                                   source_state='completed', reference=True))
    for job in manifest['jobs']:
        if job['group'] == 'structure_ablation' and job['variant'] in ('no_transfer', 'local_residual'):
            references.append(dict(job, group='repeatability', seed=0, reference=True,
                                   source_state=states[job['id']]['state']))
        if job['group'] == 'repeatability' and job['variant'] == 'full_seed1' and job['scene'] in ('Drums', 'FurScene', 'soap_small'):
            references.append(dict(job, group='training_efficiency', variant='LiSA-staged', seed=1,
                                   reference=True, source_state=states[job['id']]['state']))
    return references


def write_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def collect(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    current = json.loads(Path(manifest['status_path']).read_text())
    # The scheduler registers explicitly appended pending jobs at its next handoff.
    states = {job['id']: current['jobs'].get(job['id'], job_status_record(job))
              for job in manifest['jobs']}
    training_jobs = [job for job in manifest['jobs'] if job.get('kind', 'training') == 'training']
    measurements = [job for job in manifest['jobs'] if job.get('kind') == 'measurement']
    comparisons = [job for job in manifest['jobs'] if job.get('kind') == 'comparison']
    items = [dict(job, reference=False, source_state=states[job['id']]['state']) for job in training_jobs]
    for job in comparisons:
        for method in ('LiSA-staged', 'GS3', 'SSD-GS', 'RNG', 'OLAT-Gaussians'):
            item = dict(job, method=method, variant=method, reference=True,
                        source_state=states[job['id']]['state'],
                        resultpath=str(Path(job['output']) / f'{method}.json'))
            item.pop('id')
            items.append(item)
    items += reference_rows(manifest, states)
    rows = []
    for job in items:
        variant = job.get('variant', 'full')
        seed = int(variant.rsplit('_seed', 1)[1]) if '_seed' in variant else job.get('seed', 0)
        if job['group'] == 'repeatability':
            variant = variant.split('_seed')[0]
        row = dict(group=job['group'], variant=variant,
                   method='LiSA-staged' if job['method'] == 'PORT-GS' else job['method'],
                   family=job['family'], scene=job['scene'], seed=seed,
                   state=job['source_state'], reused=job['reference'], test_frames=None,
                   PSNR=None, SSIM=None, LPIPS=None, training_seconds=None, training_gpu_mode=None,
                   metrics_file=job['resultpath'], output=job['output'])
        if row['state'] == 'completed':
            count, values = metric_values(job['resultpath'], job['dataset'])
            row.update(test_frames=count, **values)
            if job.get('id') in states:
                row['training_seconds'] = runtime_seconds(states[job['id']])
                phases = [phase for name, phase in states[job['id']]['steps'].items()
                          if name in ('train', 'appearance')]
                row['training_gpu_mode'] = ('sharing_permitted' if any(
                    phase.get('gpu_sharing_allowed', False) for phase in phases) else 'dedicated')
        rows.append(row)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    write_csv(DESTINATION / 'results.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['group'], row['variant'], row['seed'])].append(row)
    summary = []
    for (group, variant, seed), entries in sorted(groups.items()):
        complete = [row for row in entries if row['state'] == 'completed']
        row = dict(group=group, variant=variant, seed=seed,
                   completed_scenes=len(complete), expected_scenes=len(entries))
        row.update({key: statistics.mean(item[key] for item in entries)
                    if len(complete) == len(entries) else None for key in METRICS})
        summary.append(row)
    write_csv(DESTINATION / 'summary.csv', summary)
    repeated = defaultdict(dict)
    for row in rows:
        if row['group'] == 'repeatability' and row['state'] == 'completed':
            repeated[(row['family'], row['scene'], row['variant'])][row['seed']] = row
    repeatability = []
    for (family, scene, variant), seeds in sorted(repeated.items()):
        if set(seeds) != {0, 1, 2}:
            continue
        result = dict(family=family, scene=scene, variant=variant, seeds='0,1,2')
        for key in METRICS:
            values = [seeds[seed][key] for seed in (0, 1, 2)]
            result[key + '_mean'] = statistics.mean(values)
            result[key + '_std'] = statistics.stdev(values)
        repeatability.append(result)
    if repeatability:
        write_csv(DESTINATION / 'repeatability.csv', repeatability)
    paired = []
    for family, scene, variant in sorted(repeated):
        if variant != 'full':
            continue
        full = repeated[(family, scene, variant)]
        for control in ('no_transfer', 'local_residual'):
            removed = repeated[(family, scene, control)]
            if set(full) != {0, 1, 2} or set(removed) != {0, 1, 2}:
                continue
            row = dict(family=family, scene=scene, delta='full-minus-' + control, seeds='0,1,2')
            for key in METRICS:
                values = [full[seed][key] - removed[seed][key] for seed in (0, 1, 2)]
                row[key + '_mean_delta'] = statistics.mean(values)
                row[key + '_std_delta'] = statistics.stdev(values)
            paired.append(row)
    if paired:
        write_csv(DESTINATION / 'paired_deltas.csv', paired)
    counts = {state: sum(states[job['id']]['state'] == state for job in training_jobs)
              for state in ('completed', 'running', 'pending', 'failed')}
    measurement_rows = []
    for job in measurements:
        if states[job['id']]['state'] != 'completed':
            continue
        report = json.loads(Path(job['resultpath']).read_text())
        for mode in report['modes']:
            assert all(math.isfinite(mode[key]) and mode[key] > 0
                       for key in ('fps', 'milliseconds', 'peak_allocated_MiB'))
            measurement_rows.append(dict(method=job['method'], family=job['family'], scene=job['scene'],
                                         **mode, gaussians=report['gaussians'],
                                         checkpoint_MiB=report['checkpoint_MiB'],
                                         source=job['resultpath']))
    if measurement_rows:
        write_csv(DESTINATION / 'render_efficiency.csv', measurement_rows)
    progress = dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    planned_jobs=len(training_jobs), **counts,
                    measurement_jobs=len(measurements),
                    measurements_completed=sum(states[job['id']]['state'] == 'completed' for job in measurements),
                    comparison_jobs=len(comparisons),
                    comparisons_completed=sum(states[job['id']]['state'] == 'completed' for job in comparisons),
                    manifest=str(manifest_path), status=manifest['status_path'],
                    note='Completion counts require successful phases and audited full-frame metrics. '
                         'Training time includes initialization/save and excludes separate evaluation.')
    (DESTINATION / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    print(json.dumps(progress), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    collect(parser.parse_args().manifest)

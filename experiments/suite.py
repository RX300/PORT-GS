"""Prepare the fixed LiSA supplementary panel and its baseline jobs."""

import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def replace_paths(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, replacements) for key, item in value.items()}
    return value


def baseline_jobs(config):
    jobs = []
    for method in config['baseline_experiments']['methods']:
        project = WORKSPACE / method
        destination = project / 'runs' / config['name']
        destination.mkdir(parents=True, exist_ok=False)
        archive_path = destination / 'source.tar'
        files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                                        cwd=project, text=True).splitlines()
        with tarfile.open(archive_path, 'w') as archive:
            for filename in files:
                path = project / filename
                if path.is_file() and path.suffix in ('.py', '.sh', '.json', '.md', '.cu', '.cpp', '.h'):
                    archive.add(path, arcname=filename)
        provenance = dict(
            revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
            status=subprocess.check_output(['git', 'status', '--short'], cwd=project, text=True).splitlines())
        (destination / 'source_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        templates = []
        for name in ('supplement_hard_scenes', 'full_dataset_comparison'):
            templates.extend(json.loads((project / 'runs' / name / 'jobs.json').read_text())['jobs'])
        for family, scenes in config['baseline_experiments']['scenes'].items():
            for scene in scenes:
                template = next(job for job in templates if job['family'] == family and job['scene'] == scene)
                output = destination / 'training_efficiency' / family / scene
                job = replace_paths(template, [(template['output'], str(output))])
                job.update(id=f'{method}-training_efficiency-{family}-{scene}', group='training_efficiency',
                           variant=method, state='pending', mode='fresh', exclusive_gpu=True,
                           source_archive=str(archive_path), source_revision=provenance['revision'])
                for phase in job['steps']:
                    phase['state'] = 'pending'
                    phase['output'] = str(output)
                    phase['logpath'] = str(output / 'logs' / (phase['name'] + '.log'))
                    if phase['name'] == 'train':
                        phase['progress_path'] = str(output / 'progress.jsonl')
                jobs.append(job)
        (destination / 'jobs.json').write_text(json.dumps(
            {'jobs': [job for job in jobs if job['method'] == method]}, indent=2) + '\n')
    return jobs


def measurement_jobs(config, output_dir):
    references = json.loads((WORKSPACE / 'OLAT-Gaussians/runs/full_dataset/comparison_manifest.json').read_text())['jobs']
    jobs = []
    for method in ('LiSA-staged', 'GS3', 'SSD-GS'):
        for family, scenes in config['measurement_scenes'].items():
            for scene in scenes:
                source = next(item for item in references
                              if (item['method'], item['family'], item['scene']) == (method, family, scene))
                project = ROOT if method == 'LiSA-staged' else WORKSPACE / method
                output = project / 'runs' / config['name'] / 'render_efficiency' / family / scene
                log_root = project / 'runs' / config['name'] / 'measurement_logs' / family
                if method == 'LiSA-staged':
                    checkpoint = Path(source['output']) / 'last.pt'
                    argv = [config['python'], '-B', '-m', 'experiments.measure', str(checkpoint),
                            '--output', str(output), '--ffmpeg', config['ffmpeg']]
                    env = {}
                else:
                    environment = '.venv' if method == 'GS3' else '.train-venv'
                    argv = [str(project / environment / 'bin/python'), '-B',
                            str(WORKSPACE / 'benchmarks/measure_relighting.py'), '--method', method,
                            '--model', source['output'], '--iteration', str(source['iterations']),
                            '--output', str(output / 'performance.json')]
                    env = dict(TORCH_EXTENSIONS_DIR=str(project / environment / 'torch_extensions'))
                phase = dict(name='profile', state='pending', cwd=str(project), argv=argv, env=env,
                             logpath=str(log_root / f'{scene}.profile.log'), output=str(output),
                             resultpath=str(output / 'performance.json'))
                phases = [phase]
                if method == 'LiSA-staged':
                    preview = output / 'atlas_preview'
                    phases.append(dict(name='atlas_preview', state='pending', cwd=str(ROOT),
                        argv=[config['python'], '-B', str(ROOT / 'diagnose_image_errors.py'), str(checkpoint),
                              '--light-atlas-preview', '--split', 'test', '--atlas-frames', '0', '--output', str(preview)],
                        logpath=str(log_root / f'{scene}.atlas.log'), output=str(preview),
                        resultpath=str(preview / 'preview.json')))
                jobs.append(dict(id=f'{method}-render_efficiency-{family}-{scene}', kind='measurement',
                                 exclusive_gpu=True,
                                 method=method, group='render_efficiency', variant=method,
                                 family=family, scene=scene, dataset=source['dataset'],
                                 state='pending', mode='fresh', output=str(output),
                                 source_archive=str(output_dir / 'measurement_source.tar'),
                                 resultpath=str(output / 'performance.json'), steps=phases))
    return jobs


def comparison_jobs(config, output_dir):
    panel = json.loads((WORKSPACE / 'OLAT-Gaussians/runs/full_dataset/comparison_manifest.json').read_text())
    jobs = []
    for source in panel['jobs']:
        if source['method'] != 'LiSA-staged':
            continue
        family, scene = source['family'], source['scene']
        output = output_dir / 'common_ground_truth' / family / scene
        phase = dict(name='evaluate', state='pending', cwd=str(ROOT),
                     argv=[config['python'], '-B', '-m', 'experiments.common_ground_truth',
                           '--family', family, '--scene', scene, '--output', str(output)],
                     logpath=str(output_dir / 'measurement_logs' / family / f'{scene}.common_gt.log'),
                     output=str(output), resultpath=str(output / 'audit.json'))
        jobs.append(dict(id=f'common_ground_truth-{family}-{scene}', kind='comparison',
                         method='comparison', group='common_ground_truth', variant='common_ground_truth',
                         family=family, scene=scene, dataset=source['dataset'],
                         state='pending', mode='fresh', output=str(output), resultpath=phase['resultpath'],
                         source_archive=str(output_dir / 'measurement_source.tar'), steps=[phase]))
    return jobs

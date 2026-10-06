"""Generate every LaTeX table of the paper from the stored result files (read-only).

Sources
  * common-GT re-evaluation of five methods, 18 scenes, 5,691 test images:
      runs/lisa_supplementary_experiments/common_ground_truth/<family>/<scene>/<method>.json
  * five-scene structure ablation / repeatability (3 seeds):
      docs/experiments/supplementary/{summary,repeatability,paired_deltas}.csv
  * refinement-domain study: docs/experiments/supplementary/summary.csv
  * matched 16k loss-domain study: docs/experiments/radiometric_curriculum_results.json
  * timing / rendering: docs/experiments/supplementary/{results.md,render_efficiency.csv}
    and benchmarks/full_dataset_comparison/training_times.csv
Outputs: paper/latex/tables/*.tex and paper/data/numbers.json (all numbers quoted in the text).
"""
import csv
import json
import statistics
from pathlib import Path

PORT = Path(__file__).resolve().parents[2]
WORKSPACE = PORT.parent
CGT = PORT / 'runs/lisa_supplementary_experiments/common_ground_truth'
SUPP = PORT / 'docs/experiments/supplementary'
TABLES = PORT / 'paper/latex/tables'
DATA = PORT / 'paper/data'

FAMILIES = {
    'Real_NRHints': ['Cat', 'CatSmall', 'CupFabric', 'Fish', 'FurScene', 'Pikachu', 'Pixiu'],
    'Synthetic_GS3': ['AnisoMetal', 'Drums', 'FurBall', 'Hotdog', 'Lego', 'Translucent'],
    'Synthetic_SSS-GS': ['bunny_small', 'candle_small', 'dragon_small', 'soap_small', 'statue_small'],
}
FAMILY_LABEL = {'Real_NRHints': 'NRHints (real)', 'Synthetic_GS3': 'GS$^3$ (synthetic)',
                'Synthetic_SSS-GS': 'SSS-GS (synthetic)'}
METHODS = ['RNG', 'OLAT-Gaussians', 'GS3', 'SSD-GS', 'LiSA-staged']
LABEL = {'RNG': 'RNG~\\cite{fan2025rng}', 'OLAT-Gaussians': 'OLAT-GS~\\cite{kuang2024olat}',
         'GS3': 'GS$^3$~\\cite{bi2024gs3}', 'SSD-GS': 'SSD-GS~\\cite{zheng2026ssdgs}',
         'LiSA-staged': '\\textbf{LiSA (ours)}'}
SCENE_LABEL = {'bunny_small': 'Bunny', 'candle_small': 'Candle', 'dragon_small': 'Dragon',
               'soap_small': 'Soap', 'statue_small': 'Statue'}
METRICS = ('PSNR', 'SSIM', 'LPIPS')
HIGHER = {'PSNR': True, 'SSIM': True, 'LPIPS': False}
FMT = {'PSNR': '{:.2f}', 'SSIM': '{:.3f}', 'LPIPS': '{:.3f}'}


def load_cgt():
    scores = {}
    for family, scenes in FAMILIES.items():
        for scene in scenes:
            for method in METHODS:
                report = json.loads((CGT / family / scene / f'{method}.json').read_text())
                scores[family, scene, method] = {m: report['metrics'][m] for m in METRICS}
                scores[family, scene, method]['frames'] = len(report['views'])
    return scores


def mean(values):
    return sum(values) / len(values)


def ranked_cells(values, metric, fmt=None):
    """Format a row-wise list; bold best, underline second (per column across methods)."""
    fmt = fmt or FMT[metric]
    shown = [float(fmt.format(v)) for v in values]  # rank at display precision; ties share the mark
    distinct = sorted(set(shown), reverse=HIGHER[metric])
    cells = []
    for value, rounded in zip(values, shown):
        text = fmt.format(value)
        if metric != 'PSNR':
            text = text.lstrip('0') if value < 1 else text
        if rounded == distinct[0]:
            text = f'\\textbf{{{text}}}'
        elif len(distinct) > 1 and rounded == distinct[1]:
            text = f'\\underline{{{text}}}'
        cells.append(text)
    return cells


def main_table(scores, numbers):
    groups = list(FAMILIES) + ['all']
    columns = {}
    for group in groups:
        scenes = [(f, s) for f, ss in FAMILIES.items() for s in ss if group in ('all', f)]
        for metric in METRICS:
            columns[group, metric] = [mean([scores[f, s, m][metric] for f, s in scenes]) for m in METHODS]
    numbers['main'] = {f'{g}/{metric}': dict(zip(METHODS, columns[g, metric])) for g, metric in columns}
    lines = [
        '\\begin{table*}[t]',
        '\\caption{\\textbf{Comparison on all 18 scenes of three OLAT benchmarks} (5,691 test images, identical '
        'targets, original test calibration, no test-time fitting; best in \\textbf{bold}, second \\underline{underlined}). '
        'Iterations: RNG 30k+70k, OLAT-GS 30k+30k, GS$^3$ 100k, SSD-GS 100k (60k on SSS), LiSA 38k.}',
        '\\label{tab:main}',
        '\\centering\\small',
        '\\setlength{\\tabcolsep}{4.2pt}',
        '\\begin{tabular}{@{}l ccc ccc ccc ccc@{}}',
        '\\toprule',
        ' & \\multicolumn{3}{c}{NRHints, real (7)} & \\multicolumn{3}{c}{GS$^3$, synthetic (6)} & '
        '\\multicolumn{3}{c}{SSS-GS, synthetic (5)} & \\multicolumn{3}{c}{All scenes (18)} \\\\',
        '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-13}',
        'Method' + ' & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$' * 4 + ' \\\\',
        '\\midrule',
    ]
    cells = {key: ranked_cells(values, key[1]) for key, values in columns.items()}
    for i, method in enumerate(METHODS):
        row = [LABEL[method]] + [cells[g, metric][i] for g in groups for metric in METRICS]
        if method == 'LiSA-staged':
            lines.append('\\midrule')
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    (TABLES / 'main.tex').write_text('\n'.join(lines) + '\n')


def per_scene_table(scores, numbers):
    lines = [
        '\\begin{table*}[t]',
        '\\caption{\\textbf{Per-scene results} (PSNR$\\uparrow$ / SSIM$\\uparrow$ / LPIPS$\\downarrow$), common targets, '
        'original test calibration, seed 0. Best PSNR per scene in \\textbf{bold}.}',
        '\\label{tab:perscene}',
        '\\centering\\scriptsize',
        '\\setlength{\\tabcolsep}{3pt}',
        '\\begin{tabular}{@{}ll' + 'c' * len(METHODS) + '@{}}',
        '\\toprule',
        'Benchmark & Scene & ' + ' & '.join(LABEL[m].split('~')[0] for m in METHODS) + ' \\\\',
        '\\midrule',
    ]
    wins = {m: 0 for m in METHODS}
    for family, scenes in FAMILIES.items():
        for j, scene in enumerate(scenes):
            values = [scores[family, scene, m] for m in METHODS]
            best = max(range(len(METHODS)), key=lambda i: values[i]['PSNR'])
            wins[METHODS[best]] += 1
            cells = []
            for i, v in enumerate(values):
                p = f"{v['PSNR']:.2f}"
                p = f'\\textbf{{{p}}}' if i == best else p
                cells.append(f"{p} / {v['SSIM']:.3f} / {v['LPIPS']:.3f}")
            family_cell = f'\\multirow{{{len(scenes)}}}{{*}}{{{FAMILY_LABEL[family]}}}' if j == 0 else ''
            lines.append(f'{family_cell} & {SCENE_LABEL.get(scene, scene)} & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule' if family != 'Synthetic_SSS-GS' else '\\bottomrule')
    lines += ['\\end{tabular}', '\\end{table*}']
    (TABLES / 'per_scene.tex').write_text('\n'.join(lines) + '\n')
    numbers['best_psnr_scene_counts'] = wins
    lisa_wins = {m: sum(scores[f, s, 'LiSA-staged']['PSNR'] > scores[f, s, m]['PSNR']
                        for f, ss in FAMILIES.items() for s in ss) for m in METHODS[:-1]}
    numbers['lisa_psnr_wins_vs'] = lisa_wins
    numbers['frames'] = sum(scores[f, s, 'LiSA-staged']['frames'] for f, ss in FAMILIES.items() for s in ss)
    numbers['per_scene_psnr'] = {f'{f}/{s}': {m: scores[f, s, m]['PSNR'] for m in METHODS}
                                 for f, ss in FAMILIES.items() for s in ss}


def read_csv(name):
    with (SUPP / name).open() as handle:
        return list(csv.DictReader(handle))


def ablation_table(numbers):
    summary = read_csv('summary.csv')
    rows = {(r['group'], r['variant'], r['seed']): r for r in summary}

    def seeds(variant):
        values = [rows['repeatability', variant, s] for s in '012']
        return {m: (statistics.mean(float(v[m]) for v in values), statistics.stdev(float(v[m]) for v in values))
                for m in METRICS}

    def single(variant):
        r = rows['structure_ablation', variant, '0']
        return {m: (float(r[m]), None) for m in METRICS}

    full = seeds('full')
    entries = [
        ('Full model', full, '3'),
        ('w/o transfer term', seeds('no_transfer'), '3'),
        ('transfer $\\rightarrow$ local residual$^\\dagger$', seeds('local_residual'), '3'),
        ('single atlas level ($K{=}1$)', single('single_scale'), '1'),
        ('moment test only ($g\\equiv 0$)', single('moment_visibility'), '1'),
    ]
    numbers['ablation'] = {name: {m: v for m, v in values.items()} for name, values, _ in entries}
    full0 = {m: float(rows['structure_ablation', 'full', '0'][m]) for m in METRICS}
    numbers['ablation_full_seed0'] = full0

    def cell(value, std, metric):
        fmt = FMT[metric]
        text = fmt.format(value)
        if metric != 'PSNR':
            text = text.lstrip('0')
        if std is not None and metric == 'PSNR':
            text += f'{{\\tiny$\\pm${std:.2f}}}'
        return text

    lines = [
        '\\begin{table}[t]',
        '\\caption{\\textbf{Ablation of the atlas} on five scenes (Drums, AnisoMetal, Translucent, FurScene, Soap), '
        'scene-averaged. For rows with three seeds we report the mean over seeds and the standard deviation of PSNR; single-seed rows compare to the '
        f"seed-0 full model ({full0['PSNR']:.2f}\\,dB). "
        '$^\\dagger$Capacity-matched MLP (49.1k vs.\\ 49.6k parameters) that reads the receiver\'s code, position '
        'and light/view directions but not the atlas; shadows still come from the atlas.}',
        '\\label{tab:ablation}',
        '\\centering\\small',
        '\\setlength{\\tabcolsep}{3.5pt}',
        '\\begin{tabular}{@{}lcccc@{}}',
        '\\toprule',
        'Variant & Seeds & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ \\\\',
        '\\midrule',
    ]
    for name, values, n in entries:
        lines.append(f'{name} & {n} & ' + ' & '.join(cell(*values[m], m) for m in METRICS) + ' \\\\')
    lines.append('per-Gaussian visibility$^\\ddagger$ & 3 & \\TBD & \\TBD & \\TBD \\\\')
    lines.append('hard shadow map$^\\ddagger$ & 3 & \\TBD & \\TBD & \\TBD \\\\')
    lines += ['\\bottomrule', '\\end{tabular}',
              '\\\\[2pt]{\\footnotesize$^\\ddagger$\\torun{E3 (same receivers; atlas kept for transfer); '
              'also three seeds for the single-seed rows.}}',
              '\\end{table}']
    (TABLES / 'ablation.tex').write_text('\n'.join(lines) + '\n')


def paired_table(numbers):
    deltas = read_csv('paired_deltas.csv')
    shares = json.loads((DATA / 'transfer_shares.json').read_text()) if (DATA / 'transfer_shares.json').exists() else {}
    order = [('Synthetic_GS3', 'Drums'), ('Real_NRHints', 'FurScene'), ('Synthetic_GS3', 'AnisoMetal'),
             ('Synthetic_GS3', 'Translucent'), ('Synthetic_SSS-GS', 'soap_small')]
    table = {}
    for r in deltas:
        table[r['family'], r['scene'], r['delta']] = (float(r['PSNR_mean_delta']), float(r['PSNR_std_delta']))
    numbers['paired_deltas'] = {f'{f}/{s}': {d: table[f, s, d] for d in ('full-minus-no_transfer', 'full-minus-local_residual')}
                                for f, s in order}
    lines = [
        '\\begin{table}[t]',
        '\\caption{\\textbf{Where the transfer term matters.} Per-scene PSNR differences (mean$\\pm$std of paired '
        'differences over three seeds) of the full model against removing the transfer term and against the '
        'capacity-matched local residual, and the share of linear radiance the full model routes through the '
        'transfer term (mean over 8 test views).}',
        '\\label{tab:paired}',
        '\\centering\\small',
        '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{tabular}{@{}lccc@{}}',
        '\\toprule',
        'Scene & vs.\\ w/o transfer & vs.\\ local residual & Transfer share \\\\',
        '\\midrule',
    ]
    for family, scene in order:
        a = table[family, scene, 'full-minus-no_transfer']
        b = table[family, scene, 'full-minus-local_residual']
        share = shares.get(f'{family}/{scene}')
        share_text = f"{100 * share['mean']:.0f}\\%" if share else '\\TBD'
        lines.append(f'{SCENE_LABEL.get(scene, scene)} & ${a[0]:+.2f}${{\\tiny$\\pm${a[1]:.2f}}} & '
                     f'${b[0]:+.2f}${{\\tiny$\\pm${b[1]:.2f}}} & {share_text} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (TABLES / 'paired.tex').write_text('\n'.join(lines) + '\n')


def refinement_table(numbers):
    summary = read_csv('summary.csv')
    rows = {r['variant']: r for r in summary if r['group'] == 'refinement'}
    entries = [('Gaussians', 'linear', 'gaussian_linear'), ('Gaussians', 'display', 'gaussian_observation'),
               ('pixels', 'linear', 'pixel_linear'), ('pixels', 'display', 'pixel_observation')]
    values = {m: [float(rows[key][m]) for *_, key in entries] for m in METRICS}
    numbers['refinement'] = {key: {m: float(rows[key][m]) for m in METRICS} for *_, key in entries}
    cells = {m: ranked_cells(values[m], m) for m in METRICS}
    lines = [
        '\\begin{table}[t]',
        '\\caption{\\textbf{Appearance refinement} (stage III, 8k iterations on the same frozen 30k geometry, seed 0), '
        'averaged over Lego, Drums and Soap. Our default refines pixels in the display domain.}',
        '\\label{tab:refine}',
        '\\centering\\small',
        '\\begin{tabular}{@{}llccc@{}}',
        '\\toprule',
        'Receivers & Loss domain & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ \\\\',
        '\\midrule',
    ]
    for i, (receivers, domain, _) in enumerate(entries):
        lines.append(f'{receivers} & {domain} & ' + ' & '.join(cells[m][i] for m in METRICS) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (TABLES / 'refine.tex').write_text('\n'.join(lines) + '\n')


def loss_domain_numbers(numbers):
    record = json.loads((PORT / 'docs/experiments/radiometric_curriculum_results.json').read_text())
    out = {}
    for result in record['results']:
        if 'matched_control_metrics' not in result:
            continue
        out[result['scene']] = dict(ours=result['metrics']['validation'], control=result['matched_control_metrics'],
                                    delta_psnr=result.get('delta_psnr'),
                                    per_frame=result.get('per_frame_delta') or result.get('paired_frames'))
    numbers['loss_domain'] = out


def efficiency_table(numbers):
    with (SUPP / 'render_efficiency.csv').open() as handle:
        render = [r for r in csv.DictReader(handle) if r['mode'] == 'changing_light']
    # Exclusive-GPU training minutes (supplementary results.md, LiSA = seed-1 run of the same recipe).
    train_minutes = {('Drums', 'LiSA-staged'): 25.74, ('Drums', 'GS3'): 73.53, ('Drums', 'SSD-GS'): 141.11,
                     ('FurScene', 'LiSA-staged'): 19.88, ('FurScene', 'GS3'): 50.98, ('FurScene', 'SSD-GS'): 61.17,
                     ('soap_small', 'LiSA-staged'): 16.71, ('soap_small', 'GS3'): 39.42, ('soap_small', 'SSD-GS'): 26.83}
    rows = {(r['scene'], r['method']): r for r in render}
    numbers['efficiency'] = {f'{s}/{m}': dict(train_min=train_minutes[s, m], fps=float(rows[s, m]['fps']),
                                              mem_mib=float(rows[s, m]['peak_allocated_MiB']),
                                              gaussians=int(rows[s, m]['gaussians']))
                             for s, m in train_minutes}
    lines = [
        '\\begin{table}[t]',
        '\\caption{\\textbf{Cost} on three profiled scenes, each run alone on one RTX 6000 Ada: training time '
        'in minutes (whole process, both \\ours stages) and rendering frame rate with a new light in every frame, '
        'including the light pass of \\ours.}',
        '\\label{tab:cost}',
        '\\centering\\small',
        '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{tabular}{@{}lrrrrrr@{}}',
        '\\toprule',
        ' & \\multicolumn{2}{c}{Drums} & \\multicolumn{2}{c}{FurScene} & \\multicolumn{2}{c}{Soap} \\\\',
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}',
        'Method & min & FPS & min & FPS & min & FPS \\\\',
        '\\midrule',
    ]
    for method in ('GS3', 'SSD-GS', 'LiSA-staged'):
        name = {'GS3': '\\gsthree', 'SSD-GS': 'SSD-GS', 'LiSA-staged': '\\ours'}[method]
        cells = []
        for scene in ('Drums', 'FurScene', 'soap_small'):
            cells += [f'{train_minutes[scene, method]:.1f}', f"{float(rows[scene, method]['fps']):.0f}"]
        lines.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (TABLES / 'cost.tex').write_text('\n'.join(lines) + '\n')
    # Historical all-scene wall-clock means (conditions not strictly uniform; see TRAINING_TIMES.md).
    with (WORKSPACE / 'benchmarks/full_dataset_comparison/training_times.csv').open() as handle:
        times = list(csv.DictReader(handle))
    numbers['training_times_csv_columns'] = list(times[0].keys()) if times else []


def holdout_placeholder(scores, numbers):
    """E1 placeholder on the ablation panel; the official-test column uses existing results."""
    panel = [('Synthetic_GS3', 'Drums'), ('Synthetic_GS3', 'AnisoMetal'), ('Synthetic_GS3', 'Translucent'),
             ('Real_NRHints', 'FurScene'), ('Synthetic_SSS-GS', 'soap_small')]
    with (SUPP / 'repeatability.csv').open() as handle:
        rep = {(r['scene'], r['variant']): float(r['PSNR_mean']) for r in csv.DictReader(handle)}
    official = {m: mean([scores[f, sc, m]['PSNR'] for f, sc in panel]) for m in ('GS3', 'SSD-GS', 'OLAT-Gaussians')}
    official['local'] = mean([rep[sc, 'local_residual'] for _, sc in panel])
    official['full'] = mean([rep[sc, 'full'] for _, sc in panel])
    numbers['holdout_official_panel'] = official
    rows = [('\\gsthree', official['GS3']), ('SSD-GS', official['SSD-GS']), ('OLAT-GS', official['OLAT-Gaussians']),
            ('\\ours, local residual$^\\dagger$', official['local']), ('\\ours$^\\dagger$', official['full'])]
    lines = [
        '\\begin{table}[t]',
        '\\caption{\\textbf{Held-out light directions} (\\torun{E1}). Mean PSNR over the five ablation scenes on the '
        'official test views and on training-split views whose lights fall in held-out 30$^\\circ$ angular groups, after '
        'retraining every method without those views. $^\\dagger$Three seeds.}',
        '\\label{tab:holdout}',
        '\\centering\\small',
        '\\begin{tabular}{@{}lccc@{}}',
        '\\toprule',
        'Method & Official test & Held-out lights & Drop \\\\',
        '\\midrule',
    ]
    for name, value in rows:
        lines.append(f'{name} & {value:.2f} & \\TBD & \\TBD \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (TABLES / 'holdout.tex').write_text('\n'.join(lines) + '\n')


def significance(scores, numbers):
    """Wilcoxon signed-rank tests of per-scene PSNR differences (LiSA minus baseline)."""
    from scipy.stats import wilcoxon
    out = {}
    for subset, keep in (('all', lambda f: True), ('synthetic', lambda f: f != 'Real_NRHints')):
        scenes = [(f, sc) for f, ss in FAMILIES.items() for sc in ss if keep(f)]
        for m in ('SSD-GS', 'GS3', 'OLAT-Gaussians', 'RNG'):
            d = [scores[f, sc, 'LiSA-staged']['PSNR'] - scores[f, sc, m]['PSNR'] for f, sc in scenes]
            out[f'{subset}/{m}'] = dict(p=float(wilcoxon(d).pvalue), mean_diff=mean(d),
                                        wins_2dp=sum(round(scores[f, sc, 'LiSA-staged']['PSNR'], 2) > round(scores[f, sc, m]['PSNR'], 2) for f, sc in scenes),
                                        ties_2dp=sum(round(scores[f, sc, 'LiSA-staged']['PSNR'], 2) == round(scores[f, sc, m]['PSNR'], 2) for f, sc in scenes))
        out[f'{subset}/means'] = {m: mean([scores[f, sc, m]['PSNR'] for f, sc in scenes]) for m in METHODS}
    numbers['significance'] = out


if __name__ == '__main__':
    TABLES.mkdir(parents=True, exist_ok=True)
    numbers = {}
    scores = load_cgt()
    main_table(scores, numbers)
    per_scene_table(scores, numbers)
    ablation_table(numbers)
    paired_table(numbers)
    refinement_table(numbers)
    loss_domain_numbers(numbers)
    efficiency_table(numbers)
    holdout_placeholder(scores, numbers)
    significance(scores, numbers)
    (DATA / 'numbers.json').write_text(json.dumps(numbers, indent=1) + '\n')
    print(json.dumps({k: numbers[k] for k in ('best_psnr_scene_counts', 'lisa_psnr_wins_vs', 'frames')}, indent=1))


# ---------------------------------------------------------------------------- supplement
def supp_tables():
    shares = json.loads((DATA / 'transfer_shares.json').read_text())
    novelty = json.loads((DATA / 'light_novelty.json').read_text())
    lines = ['\\begin{table}[t]', '\\caption{\\textbf{Transfer share and light novelty per scene.} Share of linear '
             'radiance routed through $L_{\\mathrm{tr}}$ by the final model (mean and range over 8 evenly spaced test '
             'views), mean learned visibility over covered pixels, and the angle between each official test light and '
             'the nearest training light (median / maximum).}', '\\label{tab:supp_shares}', '\\centering\\scriptsize',
             '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{@{}llcccc@{}}', '\\toprule',
             'Benchmark & Scene & Transfer share & Range & Mean $V$ & Test-light novelty \\\\', '\\midrule']
    for family, scenes in FAMILIES.items():
        for j, scene in enumerate(scenes):
            s = shares[f'{family}/{scene}']
            n = novelty[f'{family}/{scene}']
            first = f'\\multirow{{{len(scenes)}}}{{*}}{{{FAMILY_LABEL[family]}}}' if j == 0 else ''
            lines.append(f"{first} & {SCENE_LABEL.get(scene, scene)} & {100 * s['mean']:.1f}\\% & "
                         f"{100 * s['min']:.0f}--{100 * s['max']:.0f}\\% & {s['visibility_mean']:.2f} & "
                         f"{n['median']:.1f}$^\\circ$ / {n['max']:.1f}$^\\circ$ \\\\")
        lines.append('\\midrule' if family != 'Synthetic_SSS-GS' else '\\bottomrule')
    lines += ['\\end{tabular}', '\\end{table}']
    (TABLES / 'supp_shares.tex').write_text('\n'.join(lines) + '\n')

    with (WORKSPACE / 'benchmarks/full_dataset_comparison/summary.csv').open() as handle:
        summary = {(r['scope'], r['method']): r for r in csv.DictReader(handle)}
    versions = [('LiSA', 'Initial (30k)'), ('LiSA-v2', 'Revised (30k)'), ('LiSA-staged', 'Final (38k)')]
    scopes = [('Real_NRHints', 'Real (7)'), ('Synthetic_GS3', 'GS$^3$ (6)'), ('Synthetic_SSS-GS', 'SSS (5)'), ('all', 'All (18)')]
    numbers_history = {}
    lines = ['\\begin{table}[t]', '\\caption{\\textbf{Development history} of \\ours on the same 18-scene protocol '
             '(PSNR / LPIPS). Initial: display-domain loss and pixel receivers throughout. Revised: adds a stronger mask '
             'loss, the highlight lobes and gated background supervision. Final: the staged schedule of Sec.~4, which '
             'also changes initialization, normals and receivers.}', '\\label{tab:supp_history}',
             '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{2.5pt}', '\\begin{tabular}{@{}l' + 'c' * len(scopes) + '@{}}',
             '\\toprule', 'Version & ' + ' & '.join(label for _, label in scopes) + ' \\\\', '\\midrule']
    for method, label in versions:
        cells = []
        for scope, _ in scopes:
            r = summary[scope, method]
            cells.append(f"{float(r['PSNR']):.2f} / {float(r['LPIPS']):.3f}")
            numbers_history[f'{method}/{scope}'] = (float(r['PSNR']), float(r['LPIPS']))
        lines.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (TABLES / 'supp_history.tex').write_text('\n'.join(lines) + '\n')

    with (WORKSPACE / 'benchmarks/full_dataset_comparison/training_times.csv').open() as handle:
        times = list(csv.DictReader(handle))
    table = {(r['method'], r['family'], r['scene']): float(r['minutes']) for r in times if r['minutes']}
    order = ['LiSA-staged', 'GS3', 'SSD-GS', 'OLAT-Gaussians', 'RNG']
    lines = ['\\begin{table}[t]', '\\caption{\\textbf{Wall-clock training time} (minutes) of every run in our benchmark. '
             'Runs were not timed under strictly uniform conditions (see text); ``--\'\' marks SSD-GS runs reused '
             'without a timing log. \\Cref{tab:cost} gives exclusive-GPU measurements.}', '\\label{tab:supp_times}',
             '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{@{}l' + 'r' * len(order) + '@{}}',
             '\\toprule', 'Scene & \\ours & GS$^3$ & SSD-GS & OLAT-GS & RNG \\\\', '\\midrule']
    for family, scenes in FAMILIES.items():
        for scene in scenes:
            cells = [f'{table[m, family, scene]:.1f}' if (m, family, scene) in table else '--' for m in order]
            lines.append(f'{SCENE_LABEL.get(scene, scene)} & ' + ' & '.join(cells) + ' \\\\')
    means = []
    for m in order:
        vals = [v for (mm, f, s), v in table.items() if mm == m]
        means.append(f'{sum(vals) / len(vals):.1f}')
    lines += ['\\midrule', 'Mean & ' + ' & '.join(means) + ' \\\\', '\\bottomrule', '\\end{tabular}', '\\end{table}']
    (TABLES / 'supp_times.tex').write_text('\n'.join(lines) + '\n')
    return numbers_history


if __name__ == '__main__':
    history = supp_tables()
    numbers = json.loads((DATA / 'numbers.json').read_text())
    numbers['history'] = history
    (DATA / 'numbers.json').write_text(json.dumps(numbers, indent=1) + '\n')


def supp_registration_mask():
    path = DATA / 'registration_mask_analysis.json'
    if not path.exists():
        return {}
    analysis = json.loads(path.read_text())
    order = ['LiSA-staged', 'SSD-GS', 'GS3', 'OLAT-Gaussians', 'RNG']
    name = {'LiSA-staged': '\\ours', 'SSD-GS': 'SSD-GS', 'GS3': 'GS$^3$', 'OLAT-Gaussians': 'OLAT-GS', 'RNG': 'RNG'}
    real = [f'Real_NRHints/{s}' for s in FAMILIES['Real_NRHints']]
    out = {}
    lines = ['\\begin{table}[t]', '\\caption{\\textbf{Registration diagnostic on the real captures} (E2(b)). Mean PSNR over '
             'the seven scenes with a 24\\,px border removed, before and after translating each test rendering by the '
             'best integer 2D shift within $\\pm$24\\,px, and the mean shift length. Fixed-calibration full-image PSNR '
             'from \\cref{tab:main} for reference.}', '\\label{tab:supp_registration}', '\\centering\\small',
             '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{@{}lcccc@{}}', '\\toprule',
             'Method & Full image & Cropped & Aligned & Shift (px) \\\\', '\\midrule']
    for m in order:
        full = mean([json.loads((CGT / 'Real_NRHints' / s.split('/')[1] / f'{m}.json').read_text())['metrics']['PSNR'] for s in real])
        un = mean([analysis[s][m]['unaligned'] for s in real])
        al = mean([analysis[s][m]['aligned'] for s in real])
        sh = mean([analysis[s][m]['shift'] for s in real])
        out[m] = dict(full=full, cropped=un, aligned=al, shift=sh,
                      per_scene_aligned={s: analysis[s][m]['aligned'] for s in real})
        lines.append(f'{name[m]} & {full:.2f} & {un:.2f} & {al:.2f} & {sh:.1f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (TABLES / 'supp_registration.tex').write_text('\n'.join(lines) + '\n')

    fams = list(FAMILIES)
    lines = ['\\begin{table}[t]', '\\caption{\\textbf{Foreground-masked PSNR} (pixels with ground-truth opacity above '
             '0.5), scene-averaged per benchmark, against the common targets.}', '\\label{tab:supp_masked}',
             '\\centering\\small', '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{@{}lcccc@{}}', '\\toprule',
             'Method & Real (7) & GS$^3$ (6) & SSS (5) & All (18) \\\\', '\\midrule']
    masked = {}
    for m in order:
        cells = []
        for f in fams:
            v = mean([analysis[f'{f}/{s}'][m]['masked'] for s in FAMILIES[f]])
            masked[f'{m}/{f}'] = v
            cells.append(f'{v:.2f}')
        allv = mean([analysis[f'{f}/{s}'][m]['masked'] for f in fams for s in FAMILIES[f]])
        masked[f'{m}/all'] = allv
        lines.append(f'{name[m]} & ' + ' & '.join(cells) + f' & {allv:.2f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (TABLES / 'supp_masked.tex').write_text('\n'.join(lines) + '\n')
    return dict(registration=out, masked=masked)


if __name__ == '__main__':
    extra = supp_registration_mask()
    if extra:
        numbers = json.loads((DATA / 'numbers.json').read_text())
        numbers.update(extra)
        (DATA / 'numbers.json').write_text(json.dumps(numbers, indent=1) + '\n')

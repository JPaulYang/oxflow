"""
Compare Dice scores across interpolation methods and generate LaTeX table.

Reads dice result JSON files (complete or checkpoint) for each method,
computes summary statistics (median, mean±std, failure rate), and
runs paired one-sided Wilcoxon signed-rank tests (H_A: other > ours).

Output: prints LaTeX-ready table rows and saves detailed comparison JSON.

Usage:
    python script/eval/compare_dice.py

    # Custom paths
    python script/eval/compare_dice.py \
        --ot outputs/eval_dice_synthetic/dice_results.json \
        --linear outputs/eval_dice_synthetic_linear_mask/checkpoint_synthetic.json \
        --pixel outputs/eval_dice_synthetic_pixel_interp/checkpoint_synthetic.json

    # Include real upper bound from the OT result file
    python script/eval/compare_dice.py --include-real
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load_dice_results(json_path):
    """Load dice results from either complete or checkpoint format.

    Complete format: {"synthetic": {"results": [...]}, "real": {"results": [...]}}
    Checkpoint format: {"n_completed": N, "results": [...]}

    Returns dict with 'synthetic' and optionally 'real' as lists of result dicts.
    """
    with open(json_path) as f:
        data = json.load(f)

    out = {}

    # Complete format
    if 'synthetic' in data:
        out['synthetic'] = data['synthetic']['results']
        if 'real' in data:
            out['real'] = data['real']['results']
    # Checkpoint format
    elif 'results' in data:
        out['synthetic'] = data['results']

    return out


def compute_stats(dices):
    """Compute summary statistics for a list of dice scores."""
    arr = np.array(dices)
    return {
        'n': len(arr),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'fail_rate_50': float(np.mean(arr < 0.5) * 100),
        'fail_rate_30': float(np.mean(arr < 0.3) * 100),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
    }


def make_key(r):
    """Create a unique key for pairing samples across methods."""
    pair = tuple(r['pair']) if isinstance(r['pair'], list) else r['pair']
    return (r['patient_id'], r['slice_idx'], r['interp_t'], pair)


def paired_wilcoxon_other_greater(x_ours, x_other):
    """One-sided Wilcoxon signed-rank test: H_A: other > ours."""
    x_ours, x_other = np.array(x_ours), np.array(x_other)
    diff = x_other - x_ours
    if np.all(diff == 0):
        return 1.0
    try:
        _, p = wilcoxon(diff, alternative='greater')
        return p
    except ValueError:
        return 1.0


def paired_comparison(results_ours, results_other):
    """Pair samples by (patient_id, slice_idx, interp_t, pair) and run test.

    Returns (p_value, n_paired, win, tie, loss).
    """
    other_map = {make_key(r): r['dice'] for r in results_other}

    ours_paired, other_paired = [], []
    for r in results_ours:
        key = make_key(r)
        if key in other_map:
            ours_paired.append(r['dice'])
            other_paired.append(other_map[key])

    n_paired = len(ours_paired)
    if n_paired == 0:
        return None, 0, 0, 0, 0

    ours_arr = np.array(ours_paired)
    other_arr = np.array(other_paired)

    p_val = paired_wilcoxon_other_greater(ours_arr, other_arr)

    wins = int(np.sum(ours_arr > other_arr))
    ties = int(np.sum(ours_arr == other_arr))
    losses = int(np.sum(ours_arr < other_arr))

    return p_val, n_paired, wins, ties, losses


def format_p(p):
    """Format p-value with significance markers."""
    if p is None:
        return 'N/A'
    s = f'{p:.4f}'
    if p < 0.001:
        s = f'{p:.1e}'
    if p < 0.01:
        s += ' **'
    elif p < 0.05:
        s += ' *'
    return s


def main():
    default_ot = 'outputs/eval_dice_synthetic/dice_results.json'
    default_linear = ('outputs/eval_dice_synthetic_linear_mask/'
                      'checkpoint_synthetic.json')
    default_pixel = ('outputs/eval_dice_synthetic_pixel_interp/'
                     'checkpoint_synthetic.json')
    default_random = ('outputs/eval_dice_random_mask_bash/'
                      'checkpoint_synthetic.json')

    parser = argparse.ArgumentParser(
        description='Compare Dice scores and generate LaTeX table')
    parser.add_argument('--ot', type=str, default=default_ot,
                        help='OT method result JSON')
    parser.add_argument('--linear', type=str, default=default_linear,
                        help='Linear mask method result JSON')
    parser.add_argument('--pixel', type=str, default=default_pixel,
                        help='Pixel interpolation result JSON')
    parser.add_argument('--random', type=str, default=default_random,
                        help='Random mask method result JSON')
    parser.add_argument('--include-real', action='store_true',
                        help='Include real data upper bound from OT file')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/comparison',
                        help='Directory to save comparison results')
    args = parser.parse_args()

    # Load results
    methods = {}
    method_order = []

    if args.include_real:
        ot_data = load_dice_results(args.ot)
        if 'real' in ot_data:
            methods['Real (upper bound)'] = ot_data['real']
            method_order.append('Real (upper bound)')

    for name, path in [('OT mask + FM (Ours)', args.ot),
                       ('Linear mask + FM', args.linear),
                       ('Pixel interpolation', args.pixel),
                       ('Random mask + FM', args.random)]:
        p = Path(path)
        if p.exists():
            data = load_dice_results(path)
            methods[name] = data['synthetic']
            method_order.append(name)
        else:
            print(f"Warning: {path} not found, skipping {name}")

    if not methods:
        print("Error: no result files found.")
        return

    ours_name = 'OT mask + FM (Ours)'
    ours_results = methods.get(ours_name, None)

    # Compute stats and comparisons
    print(f"\n{'='*80}")
    print("Dice Score Comparison")
    print(f"{'='*80}\n")

    header = (f"{'Method':<25} {'Median':>8} {'Mean±Std':>14} "
              f"{'Dice<0.5(%)':>12} {'n':>7} {'p-value':>14} {'W/T/L':>15}")
    print(header)
    print('-' * len(header))

    json_data = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'methods': {},
    }

    for name in method_order:
        results = methods[name]
        dices = [r['dice'] for r in results]
        stats = compute_stats(dices)

        # Paired test vs Ours
        p_val, n_paired, wins, ties, losses = None, 0, 0, 0, 0
        if name == ours_name:
            p_str = '(ref)'
            wtl_str = '--'
        elif name == 'Real (upper bound)':
            p_str = '--'
            wtl_str = '--'
        elif ours_results is not None:
            p_val, n_paired, wins, ties, losses = paired_comparison(
                ours_results, results)
            p_str = format_p(p_val)
            wtl_str = f'{wins}/{ties}/{losses}'
        else:
            p_str = 'N/A'
            wtl_str = 'N/A'

        print(f"{name:<25} {stats['median']:>8.4f} "
              f"{stats['mean']:.4f}±{stats['std']:.4f} "
              f"{stats['fail_rate_50']:>11.1f} "
              f"{stats['n']:>7} "
              f"{p_str:>14} {wtl_str:>15}")

        json_data['methods'][name] = {
            **stats,
            'p_value_vs_ours': float(p_val) if p_val is not None else None,
            'n_paired': n_paired,
            'wins': wins,
            'ties': ties,
            'losses': losses,
        }

    # LaTeX table
    print(f"\n{'='*80}")
    print("LaTeX Table Rows (copy-paste into paper)")
    print(f"{'='*80}\n")

    print(r"\begin{table}[t]")
    print(r"\caption{Anatomical plausibility: Dice overlap between "
          r"conditioning mask and nnU-Net segmentation prediction on "
          r"synthetic images.}\label{tab:dice}")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Method & Median$\uparrow$ & Mean $\pm$ Std & "
          r"Fail (\%)$\downarrow$ \\")
    print(r"\hline")

    for name in method_order:
        stats = json_data['methods'][name]
        median_str = f"{stats['median']:.3f}"
        mean_std_str = f"{stats['mean']:.3f} $\\pm$ {stats['std']:.3f}"

        fail_str = f"{stats['fail_rate_50']:.1f}"

        # Bold the best synthetic method (exclude real upper bound)
        is_ours = (name == ours_name)
        if is_ours:
            median_str = r"\textbf{" + median_str + "}"
            mean_std_str = (r"\textbf{" +
                            f"{stats['mean']:.3f}" + r"}" +
                            f" $\\pm$ {stats['std']:.3f}")
            fail_str = r"\textbf{" + fail_str + "}"

        # Add hline before synthetic methods
        if name == ours_name and 'Real (upper bound)' in method_order:
            print(r"\hline")

        print(f"{name} & {median_str} & {mean_std_str} & "
              f"{fail_str} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Per-interp_t breakdown
    print(f"\n{'='*80}")
    print("Breakdown by interpolation t")
    print(f"{'='*80}\n")

    t_header = f"{'Method':<25}"
    t_values = sorted({r['interp_t']
                       for name in method_order
                       for r in methods[name]
                       if 'interp_t' in r})
    for t in t_values:
        t_header += f"  t={t:.2f}"
    print(t_header)
    print('-' * len(t_header))

    for name in method_order:
        results = methods[name]
        by_t = {}
        for r in results:
            if 'interp_t' in r:
                t = r['interp_t']
                by_t.setdefault(t, []).append(r['dice'])
        if not by_t:
            continue
        row = f"{name:<25}"
        for t in t_values:
            if t in by_t:
                row += f"  {np.median(by_t[t]):>.4f}"
            else:
                row += "     --"
        print(row)

    # Save JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f"dice_comparison_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved to: {json_path}")


if __name__ == '__main__':
    main()

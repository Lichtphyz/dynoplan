"""
Cost and path length over time: iDB-A*, SST*, RRT, and DiTree (4 algorithms).
Based on plot_paper_figure.py but adds DiTree from DP_data CSVs.

DiTree data format (DP_data/*.csv):
  iteration, success, runtime, trajectory_length, trajectory_time, ...
  - runtime: wall time to first (and only) solution [s]
  - trajectory_time: cost / travel time [s]
  - trajectory_length: path length [m]
  - success=0 or runtime > TLIMIT treated as unsolved within window

Included envs: Zig, Loop, Whole, narrow_short, open_blocks_small.
'Race Track_diffusion_RRT_PD64_carmaze.csv' maps to Race_Track_Whole.
"""
import glob
import numpy as np
import yaml
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

BASE   = '/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan/results_new/car_dyn_v0_acc120'
STAMP  = '2026-03-21--13-49-53'
DP_DIR = '/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan/figures/DP_data'
TLIMIT = 60
N_TIMES = 500

ENVS = ['Race_Track_Zig', 'Race_Track_Loop', 'Race_Track_Whole', 'narrow_short', 'open_blocks_small']
ENV_LABELS = {
    'Race_Track_Zig':    'Race Track Zig',
    'Race_Track_Loop':   'Race Track Loop',
    'Race_Track_Whole':  'Race Track Whole',
    'narrow_short':      'Narrow Short',
    'open_blocks_small': 'Open Blocks Small',
}

# Mapping from dynoplan env name to DiTree CSV filename stem
DP_FILE = {
    'Race_Track_Zig':    'Race Track Zig_diffusion_RRT_PD64_carmaze.csv',
    'Race_Track_Loop':   'Race Track Loop_diffusion_RRT_PD64_carmaze.csv',
    'Race_Track_Whole':  'Race Track_diffusion_RRT_PD64_carmaze.csv',
    'narrow_short':      'narrow_diffusion_RRT_PD64_carmaze.csv',
    'open_blocks_small': 'open_blocks_small_diffusion_RRT_PD64_carmaze.csv',
}

DYNO_ALGS = ['sst_rrt_v0', 'sst_v0', 'idbastar_v0']
ALG_LABELS = {
    'sst_rrt_v0': 'RRT',
    'sst_v0':     'SST*',
    'idbastar_v0':'iDB-A*',
    'ditree':     'DiTree',
}
ALG_COLORS = {
    'sst_rrt_v0': 'steelblue',
    'sst_v0':     'darkorange',
    'idbastar_v0':'forestgreen',
    'ditree':     'crimson',
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _path_length(states):
    total = 0.0
    for i in range(len(states) - 1):
        dx = states[i+1][0] - states[i][0]
        dy = states[i+1][1] - states[i][1]
        total += (dx**2 + dy**2)**0.5
    return total


def load_dyno_series(filepath):
    """Return (cost_pairs, length_pairs) as monotone-improving (time_s, value) lists."""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    trajs = data.get('trajs_opt') or []
    cost_pairs, len_pairs = [], []
    best_cost, best_len = 1e18, 1e18
    for t in trajs:
        if not t.get('feasible'):
            continue
        states = t.get('states') or []
        if len(states) < 2:
            continue
        tts    = float(t.get('time_stamp', 0)) / 1000.0
        cost   = float(t.get('cost', 1e18))
        length = _path_length(states)
        if cost < best_cost:
            cost_pairs.append((tts, cost))
            best_cost = cost
        if length < best_len:
            len_pairs.append((tts, length))
            best_len = length
    return (cost_pairs or None), (len_pairs or None)


def load_ditree(env):
    """Return list of (runtime, cost, length) per trial. Unsolved = success==0."""
    path = f'{DP_DIR}/{DP_FILE[env]}'
    trials = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            success = int(row['success'])
            runtime = float(row['runtime'])
            cost    = float(row['trajectory_time'])   if success else None
            length  = float(row['trajectory_length']) if success else None
            trials.append((runtime, cost, length))
    return trials


def interpolate(pairs, times):
    ts = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]
    ts_ext = [0.0] + ts + [times[-1]]
    vs_ext = [np.nan] + vs + [vs[-1]]
    f = interp1d(ts_ext, vs_ext, kind='previous')
    return f(times)


def ditree_series(trials, metric, times):
    """Build interpolated series for DiTree (single solution per trial)."""
    # metric: 'cost' or 'length'
    idx = 1 if metric == 'cost' else 2
    all_series = []
    for runtime, cost, length in trials:
        val = cost if metric == 'cost' else length
        if val is None or runtime > TLIMIT:
            all_series.append(np.full(len(times), np.nan))
        else:
            all_series.append(interpolate([(runtime, val)], times))
    return all_series


def median_band(all_series):
    arr = np.array(all_series, dtype=float)
    with np.errstate(all='ignore'):
        med = np.nanmedian(arr, axis=0)
        p20 = np.nanpercentile(arr, 20, axis=0)
        p80 = np.nanpercentile(arr, 80, axis=0)
    return med, p20, p80


def plot_col(ax, env, metric, times):
    any_data = False

    # Dynoplan algorithms
    for alg in DYNO_ALGS:
        alg_dir = f'{BASE}/{env}/{alg}/{STAMP}'
        files = sorted(glob.glob(f'{alg_dir}/run_*_out.yaml'))
        if not files:
            continue
        all_series = []
        for fp in files:
            cost_p, len_p = load_dyno_series(fp)
            pairs = cost_p if metric == 'cost' else len_p
            if pairs is None:
                all_series.append(np.full(len(times), np.nan))
            else:
                all_series.append(interpolate(pairs, times))
        med, p20, p80 = median_band(all_series)
        color = ALG_COLORS[alg]
        n_solved = sum(1 for s in all_series if not np.all(np.isnan(s)))
        ax.step(times, med, where='post', color=color, linewidth=1.5,
                label=f'{ALG_LABELS[alg]} ({n_solved}/{len(files)})')
        ax.fill_between(times, p20, p80, step='post', alpha=0.2, color=color)
        any_data = True

    # DiTree
    trials = load_ditree(env)
    all_series = ditree_series(trials, metric, times)
    med, p20, p80 = median_band(all_series)
    color = ALG_COLORS['ditree']
    n_solved = sum(1 for (runtime, c, l) in trials
                   if c is not None and runtime <= TLIMIT)
    ax.step(times, med, where='post', color=color, linewidth=1.5,
            label=f'{ALG_LABELS["ditree"]} ({n_solved}/{len(trials)})')
    ax.fill_between(times, p20, p80, step='post', alpha=0.2, color=color)
    any_data = True

    ax.set_xlim(0, TLIMIT)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if any_data:
        ax.legend(fontsize=6, loc='upper right')


# ── layout ───────────────────────────────────────────────────────────────────

n_rows = len(ENVS)
fig, axes = plt.subplots(n_rows, 2, figsize=(7, 1.8 * n_rows))
times = np.linspace(0, TLIMIT, N_TIMES)

for row, env in enumerate(ENVS):
    ax_cost = axes[row, 0]
    ax_len  = axes[row, 1]

    plot_col(ax_cost, env, 'cost',   times)
    plot_col(ax_len,  env, 'length', times)

    ax_cost.set_ylabel(ENV_LABELS[env], fontsize=8, labelpad=4)
    if row == 0:
        ax_cost.set_title('Cost [s]', fontsize=9, fontweight='bold')
        ax_len.set_title('Path Length [m]', fontsize=9, fontweight='bold')
    if row == n_rows - 1:
        ax_cost.set_xlabel('time [s]', fontsize=8)
        ax_len.set_xlabel('time [s]', fontsize=8)

plt.tight_layout(h_pad=0.6, w_pad=1.0)

out = '/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan/figures/cost_dist_over_time_figure_4alg.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')

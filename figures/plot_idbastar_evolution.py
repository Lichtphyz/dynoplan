"""
iDB-A* path evolution figure for paper.
All 5 environments, single seed each.
Per env: top row = cost-over-time curve with snapshot markers;
         bottom row = path snapshots side by side (oldest→best).
"""
import glob
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

RESULTS_BASE = '/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan/results_new/car_dyn_v0_acc120'
ENV_BASE     = '/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan/dynobench/envs/car_dyn_v0_acc120'
STAMP        = '2026-03-21--13-49-53'
TLIMIT       = 20

ENVS = [
    ('Race_Track_Zig',   'run_0_out.yaml',  'Race Track Zig'),
    ('narrow_short',     'run_7_out.yaml',  'Narrow Short'),
    ('open_blocks_small','run_4_out.yaml',  'Open Blocks Small'),
]

# Colormap: earliest solution light, best (last) solution dark
SNAP_CMAP = plt.cm.plasma
SNAP_ALPHA_PATH = 0.9


# ── helpers ──────────────────────────────────────────────────────────────────

def load_env(env_name):
    path = f'{ENV_BASE}/{env_name}.yaml'
    with open(path) as f:
        return yaml.safe_load(f)


def load_improvements(filepath):
    """Return list of dicts with keys: time_s, cost, states — monotone cost-improving."""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    trajs = data.get('trajs_opt') or []
    results = []
    best = 1e18
    for t in trajs:
        if not t.get('feasible'):
            continue
        states = t.get('states') or []
        if len(states) < 2:
            continue
        cost = float(t.get('cost', 1e18))
        tts  = float(t.get('time_stamp', 0)) / 1000.0
        if cost < best:
            results.append({'time_s': tts, 'cost': cost, 'states': states})
            best = cost
    return results


def draw_obstacles(ax, env_data):
    for obs in env_data['environment'].get('obstacles', []):
        if obs['type'] == 'box':
            cx, cy = obs['center']
            sx, sy = obs['size']
            rect = patches.Rectangle(
                (cx - sx/2, cy - sy/2), sx, sy,
                linewidth=0, facecolor='#cccccc', zorder=1)
            ax.add_patch(rect)
    emin = env_data['environment']['min']
    emax = env_data['environment']['max']
    ax.set_xlim(emin[0], emax[0])
    ax.set_ylim(emin[1], emax[1])
    ax.set_aspect('equal')
    ax.axis('off')


def draw_path(ax, states, color, lw=1.5, zorder=2):
    xs = [s[0] for s in states]
    ys = [s[1] for s in states]
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder, solid_capstyle='round')
    # Mark start and goal
    ax.plot(xs[0],  ys[0],  'o', color=color, markersize=4, zorder=zorder+1)
    ax.plot(xs[-1], ys[-1], 's', color=color, markersize=4, zorder=zorder+1)


# ── layout ───────────────────────────────────────────────────────────────────

n_envs = len(ENVS)
fig = plt.figure(figsize=(3.5, 3.0 * n_envs))
outer = gridspec.GridSpec(n_envs, 1, figure=fig, hspace=0.1)

for env_idx, (env_name, run_file, env_label) in enumerate(ENVS):
    env_data     = load_env(env_name)
    run_path     = f'{RESULTS_BASE}/{env_name}/idbastar_v0/{STAMP}/{run_file}'
    improvements = load_improvements(run_path)

    n_snaps = len(improvements)
    colors  = [SNAP_CMAP(0.15 + 0.7 * i / max(n_snaps - 1, 1)) for i in range(n_snaps)]

    # Inner grid: row 0 = cost curve (full width), row 1 = path snapshots
    inner = gridspec.GridSpecFromSubplotSpec(
        2, n_snaps, subplot_spec=outer[env_idx],
        height_ratios=[1, 2.0], hspace=0.45, wspace=0.06)

    # ── cost curve (spans all columns) ────────────────────────────────────────
    ax_cost = fig.add_subplot(inner[0, :])

    # Step-plot of cost over time using all feasible trajs (including non-improving)
    all_run_path = f'{RESULTS_BASE}/{env_name}/idbastar_v0/{STAMP}/{run_file}'
    with open(all_run_path) as f:
        raw = yaml.safe_load(f)
    all_trajs = raw.get('trajs_opt') or []
    curve_t, curve_c = [0.0], [None]
    best = 1e18
    for t in all_trajs:
        if not t.get('feasible') or len(t.get('states') or []) < 2:
            continue
        c   = float(t.get('cost', 1e18))
        tts = float(t.get('time_stamp', 0)) / 1000.0
        if c < best:
            curve_t.append(tts)
            curve_c.append(c)
            best = c
    if len(curve_t) > 1:
        curve_t.append(TLIMIT)
        curve_c.append(curve_c[-1])
        # Draw step line, skipping the initial None
        ax_cost.step(curve_t[1:], curve_c[1:], where='post',
                     color='forestgreen', linewidth=1.5)

    # Markers for each snapshot
    for i, imp in enumerate(improvements):
        ax_cost.axvline(imp['time_s'], color=colors[i], linewidth=0.9,
                        linestyle='--', alpha=0.7, zorder=3)
        ax_cost.plot(imp['time_s'], imp['cost'], 'o', color=colors[i],
                     markersize=5, zorder=4)

    ax_cost.set_xlim(0, TLIMIT)
    ax_cost.set_ylim(bottom=0)
    ax_cost.set_xlabel('time [s]', fontsize=7)
    ax_cost.set_ylabel('cost [s]', fontsize=7)
    ax_cost.set_title(env_label, fontsize=8, fontweight='bold')
    ax_cost.tick_params(labelsize=6)
    ax_cost.grid(True, alpha=0.3)

    # ── path snapshots ─────────────────────────────────────────────────────────
    for i, imp in enumerate(improvements):
        ax_p = fig.add_subplot(inner[1, i])
        draw_obstacles(ax_p, env_data)
        draw_path(ax_p, imp['states'], color=colors[i], lw=1.3)
        ax_p.set_title(f't={imp["time_s"]:.1f}s\nc={imp["cost"]:.2f}',
                       fontsize=6, pad=2)

out = '/home/dlicht/SCU/MotionPlanning/FinalProject/figures/idbastar_evolution.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')

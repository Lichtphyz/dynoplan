"""
Post-processing for car_dyn_v0_acc120 benchmark (stamp 2026-03-21--13-49-53):
  1. Run analyze_runs for all env/alg combos
  2. Generate comparison PDF
  3. Generate trajectory overview PNG (5 envs x 3 algs)

Run from any directory:
    python3 /home/dlicht/SCU/MotionPlanning/FinalProject/tmp/post_bench_acc120.py
"""
import sys, os, glob
sys.path.insert(0, '/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan/benchmark')
sys.path.insert(0, '/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan')
os.chdir('/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan/buildRelease')
import matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import benchmark

BASE    = '/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan/results_new/car_dyn_v0_acc120'
ENV_DIR = '/home/dlicht/SCU/MotionPlanning/FinalProject/dynoplan/dynobench/envs/car_dyn_v0_acc120'
TMP     = '/home/dlicht/SCU/MotionPlanning/FinalProject/figures'
TIMESTAMP = '2026-03-21--13-49-53'

ENVS = ['Race_Track_Zig', 'Race_Track_Loop', 'Race_Track_Whole', 'narrow_short', 'open_blocks_small']
ALGS = ['idbastar_v0', 'sst_v0', 'sst_rrt_v0']
ALG_LABELS = {'idbastar_v0': 'iDB-A*', 'sst_v0': 'SST*', 'sst_rrt_v0': 'RRT'}
ALG_COLORS = {'idbastar_v0': 'forestgreen', 'sst_v0': 'darkorange', 'sst_rrt_v0': 'steelblue'}
ENV_LABELS = {
    'Race_Track_Zig':    'Race Track Zig',
    'Race_Track_Loop':   'Race Track Loop',
    'Race_Track_Whole':  'Race Track Whole',
    'narrow_short':      'Narrow Short',
    'open_blocks_small': 'Open Blocks Small',
}

N_TRIALS = len(glob.glob(f'{BASE}/{ENVS[0]}/{ALGS[0]}/{TIMESTAMP}/run_*_out.yaml'))
print(f'Timestamp: {TIMESTAMP}')
print(f'Trials detected: {N_TRIALS}')

# ---------------------------------------------------------------------------
# 1. Run analyze_runs for all env/alg combos
# ---------------------------------------------------------------------------
print('\n=== Generating per-alg reports ===')
report_files = []
for env in ENVS:
    for alg in ALGS:
        path = f'{BASE}/{env}/{alg}/{TIMESTAMP}'
        problem = f'car_dyn_v0_acc120/{env}'
        print(f'  {env}/{alg}')
        try:
            fileout, _ = benchmark.analyze_runs(path, problem, alg, visualize=False)
            report_files.append(fileout)
        except Exception as e:
            print(f'    ERROR: {e}')

# ---------------------------------------------------------------------------
# 2. Generate comparison PDF + summary CSV
# ---------------------------------------------------------------------------
print('\n=== Generating comparison PDF ===')
try:
    benchmark.compare(report_files)
    print('  OK')
except Exception as e:
    print(f'  ERROR: {e}')

# ---------------------------------------------------------------------------
# 3. Trajectory overview PNG
# ---------------------------------------------------------------------------
def draw_env(ax, env_data):
    env = env_data['environment']
    mn, mx = env['min'], env['max']
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_facecolor('#f5f5f5')
    for obs in env.get('obstacles', []):
        cx, cy = obs['center']
        sx, sy = obs['size']
        rect = patches.Rectangle((cx - sx/2, cy - sy/2), sx, sy,
                                  linewidth=0, facecolor='#555555')
        ax.add_patch(rect)

def load_paths(env, alg):
    paths = []
    for f in sorted(glob.glob(f'{BASE}/{env}/{alg}/{TIMESTAMP}/run_*_out.yaml')):
        with open(f) as fp:
            d = yaml.safe_load(fp)
        trajs = d.get('trajs_opt') or []
        best_cost, best_states = 1e9, None
        for t in trajs:
            if t.get('feasible') and t.get('cost', 1e9) < best_cost:
                s = t.get('states') or []
                if len(s) >= 2:
                    best_cost, best_states = t['cost'], s
        if best_states:
            paths.append(best_states)
    return paths

print('\n=== Generating trajectory overview ===')
matplotlib.rcParams['text.usetex'] = False
env_data_cache = {}
for env in ENVS:
    with open(f'{ENV_DIR}/{env}.yaml') as fp:
        env_data_cache[env] = yaml.safe_load(fp)

print('Loading paths...')
path_cache = {}
for env in ENVS:
    for alg in ALGS:
        path_cache[(env, alg)] = load_paths(env, alg)
        print(f'  {env}/{alg}: {len(path_cache[(env, alg)])} paths')

print('Plotting...')
fig, axes = plt.subplots(len(ENVS), len(ALGS), figsize=(len(ALGS)*5, len(ENVS)*4))

for ri, env in enumerate(ENVS):
    ed = env_data_cache[env]
    robot = ed.get('robots', [{}])[0]
    start = robot.get('start', [])
    goal  = robot.get('goal', [])

    for ci, alg in enumerate(ALGS):
        ax = axes[ri][ci]
        draw_env(ax, ed)

        color = ALG_COLORS[alg]
        paths = path_cache[(env, alg)]
        for states in paths:
            xs = [s[0] for s in states]
            ys = [s[1] for s in states]
            ax.plot(xs, ys, '-', color=color, linewidth=0.8, alpha=0.3)

        if len(start) >= 2:
            ax.plot(start[0], start[1], 'g^', markersize=8, zorder=5)
        if len(goal) >= 2:
            ax.plot(goal[0], goal[1], 'rs', markersize=8, zorder=5)

        n = len(paths)
        ax.set_title(f'{ENV_LABELS[env]} — {ALG_LABELS[alg]}\n{n}/{N_TRIALS} solved', fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(f'car_dyn_v0_acc120 - All Trajectories (alpha=0.3, start=^, goal=sq)\n{TIMESTAMP}',
             fontsize=13, fontweight='bold')
plt.tight_layout()

out_png = f'{TMP}/trajs_acc120_{TIMESTAMP}.png'
plt.savefig(out_png, dpi=130, bbox_inches='tight')
plt.close()
print(f'Saved: {out_png}')
print('\nAll done.')

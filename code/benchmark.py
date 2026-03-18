"""
Benchmark orchestrator for the Drosophila brain model.

Manages shared configuration, logging, CSV result persistence, and dispatches
to framework-specific runners:
  - run_brian2_cuda.py  (Brian2 C++ standalone / Brian2CUDA)
  - run_pytorch.py      (PyTorch)
  - run_nestgpu.py      (NEST GPU)

Entrypoint is in main.py at the project root.
"""

import os
import csv
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

os.environ['PYTHONUNBUFFERED'] = '1'

from pathlib import Path
from datetime import datetime

# ============================================================================
# Benchmark Configuration
# ============================================================================

T_RUN_VALUES_SEC = [0.1, 1, 10, 100, 1000]
N_RUN_VALUES = [1, 30]

# ============================================================================
# Paths and Constants
# ============================================================================
current_dir = Path(__file__).resolve().parent
output_dir = current_dir / 'output'
path_comp = (current_dir / '../data/2025_Completeness_783.csv').resolve()
path_con = (current_dir / '../data/2025_Connectivity_783.parquet').resolve()
path_res = (current_dir / '../data/results').resolve()
path_wt = (current_dir / '../data').resolve()
csv_path = (current_dir / '../data/benchmark-results.csv').resolve()

# ============================================================================
# Experiment Definitions
# ============================================================================

EXPERIMENTS = {
    'sugar': {
        'key': 'sugar',
        'name': 'Sugar GRNs (200 Hz)',
        'neu_exc': [
            720575940624963786,
            720575940630233916,
            720575940637568838,
            720575940638202345,
            720575940617000768,
            720575940630797113,
            720575940632889389,
            720575940621754367,
            720575940621502051,
            720575940640649691,
            720575940639332736,
            720575940616885538,
            720575940639198653,
            720575940639259967,
            720575940617937543,
            720575940632425919,
            720575940633143833,
            720575940612670570,
            720575940628853239,
            720575940629176663,
            720575940611875570,
        ],
        'neu_exc2': [],
        'neu_slnc': [],
        'stim_rate': 200.0,
    },
    'p9': {
        'key': 'p9',
        'name': 'P9s forward walking (100 Hz)',
        'neu_exc': [
            720575940627652358,  # P9 left
            720575940635872101,  # P9 right
        ],
        'neu_exc2': [],
        'neu_slnc': [],
        'stim_rate': 100.0,
    },
    'whole_brain': {
        'key': 'whole_brain',
        'name': 'Whole Brain (300 hubs+inputs, 200 Hz)',
        'neu_exc': [
            720575940602953388, 720575940603511584, 720575940603605024,
            720575940603776702, 720575940605057922, 720575940605417696,
            720575940605623776, 720575940605663148, 720575940605680300,
            720575940605746348, 720575940605758142, 720575940605758688,
            720575940605767084, 720575940605824684, 720575940605891756,
            720575940605892012, 720575940605926112, 720575940605953982,
            720575940605969324, 720575940605981100, 720575940605983660,
            720575940605983916, 720575940605989804, 720575940606019312,
            720575940606023408, 720575940606093808, 720575940606192318,
            720575940606261792, 720575940606278688, 720575940606295486,
            720575940606303755, 720575940606305056, 720575940606315040,
            720575940606331326, 720575940606367422, 720575940606401982,
            720575940606450353, 720575940606453950, 720575940606459424,
            720575940606460606, 720575940606475366, 720575940606492774,
            720575940606504224, 720575940606515558, 720575940606518206,
            720575940606519910, 720575940606523494, 720575940606530406,
            720575940606534833, 720575940606537429, 720575940606566321,
            720575940606567089, 720575940606574694, 720575940606595686,
            720575940606628286, 720575940606629822, 720575940606630176,
            720575940606640230, 720575940606648608, 720575940606666016,
            720575940606682558, 720575940606692542, 720575940606714801,
            720575940606744241, 720575940606756030, 720575940606851249,
            720575940606923441, 720575940607168457, 720575940607479497,
            720575940607550921, 720575940607635913, 720575940607759561,
            720575940607927170, 720575940607967874, 720575940608016514,
            720575940608022194, 720575940608025218, 720575940608033666,
            720575940608056962, 720575940608139442, 720575940608152450,
            720575940608181122, 720575940608221618, 720575940608235186,
            720575940608244658, 720575940608545219, 720575940608945500,
            720575940609023068, 720575940609045257, 720575940609137500,
            720575940609157074, 720575940609187101, 720575940609229660,
            720575940609229835, 720575940609259356, 720575940609270793,
            720575940609282825, 720575940609310300, 720575940609319177,
            720575940609329929, 720575940609362187, 720575940609363292,
            720575940609442569, 720575940609456395, 720575940609488905,
            720575940609496073, 720575940609525769, 720575940609647729,
            720575940609657355, 720575940609688329, 720575940610640482,
            720575940611090244, 720575940611563310, 720575940611719378,
            720575940611772514, 720575940612264817, 720575940612718563,
            720575940612976497, 720575940613065130, 720575940613493733,
            720575940613583001, 720575940613588246, 720575940613635737,
            720575940613686698, 720575940613891442, 720575940614137809,
            720575940614363363, 720575940614710509, 720575940615505298,
            720575940615743650, 720575940616534338, 720575940617528693,
            720575940618153105, 720575940618165393, 720575940618229051,
            720575940618444473, 720575940618554041, 720575940618851073,
            720575940619055617, 720575940619073620, 720575940619637056,
            720575940619811638, 720575940619991862, 720575940620008758,
            720575940620065694, 720575940620330801, 720575940620609739,
            720575940620781515, 720575940620827595, 720575940620975696,
            720575940621103239, 720575940621116807, 720575940621148948,
            720575940621371045, 720575940621393172, 720575940621522624,
            720575940621619627, 720575940621638534, 720575940621800539,
            720575940621801819, 720575940622160705, 720575940622313398,
            720575940622331830, 720575940622440308, 720575940622516852,
            720575940622523508, 720575940622719770, 720575940622729087,
            720575940623000858, 720575940623072013, 720575940623133097,
            720575940623149089, 720575940623167838, 720575940623432918,
            720575940623444044, 720575940623475324, 720575940623499132,
            720575940623636701, 720575940623788040, 720575940623877475,
            720575940623888397, 720575940624225741, 720575940624258857,
            720575940624313834, 720575940624331144, 720575940624338056,
            720575940624394790, 720575940624402173, 720575940624411283,
            720575940624477587, 720575940624537578, 720575940624547622,
            720575940624774839, 720575940625000200, 720575940625049781,
            720575940625102224, 720575940625357547, 720575940625403144,
            720575940625525740, 720575940625550823, 720575940625741287,
            720575940625926693, 720575940625934094, 720575940625952755,
            720575940626000056, 720575940626039996, 720575940626044942,
            720575940626403917, 720575940626462014, 720575940626872932,
            720575940626979621, 720575940627190556, 720575940627247802,
            720575940627250238, 720575940627277242, 720575940627277754,
            720575940627344740, 720575940627497244, 720575940627502338,
            720575940627594568, 720575940627706398, 720575940627796298,
            720575940627996285, 720575940628038808, 720575940628069501,
            720575940628082053, 720575940628307026, 720575940628346634,
            720575940628360506, 720575940628406538, 720575940628443962,
            720575940628731140, 720575940628762280, 720575940628880808,
            720575940628908548, 720575940629382150, 720575940629486714,
            720575940629494022, 720575940629513222, 720575940629614953,
            720575940629959759, 720575940630460919, 720575940630568830,
            720575940630810306, 720575940631147776, 720575940631149532,
            720575940631180599, 720575940631190016, 720575940631306284,
            720575940631584876, 720575940631608313, 720575940631758393,
            720575940631945815, 720575940632069583, 720575940632311115,
            720575940632403986, 720575940632504874, 720575940632689199,
            720575940632777320, 720575940632795532, 720575940632821352,
            720575940632891961, 720575940633176204, 720575940633305681,
            720575940633573836, 720575940633631396, 720575940634545890,
            720575940634612194, 720575940634638562, 720575940635089945,
            720575940635188892, 720575940635468991, 720575940635684762,
            720575940636479668, 720575940636489140, 720575940636543668,
            720575940636611576, 720575940637098352, 720575940637689486,
            720575940638128474, 720575940638720233, 720575940638861631,
            720575940639172585, 720575940639209956, 720575940639823413,
            720575940640062005, 720575940640141757, 720575940640263989,
            720575940640736371, 720575940640803200, 720575940640851363,
            720575940643255693, 720575940644702112, 720575940645693988,
            720575940647182371, 720575940648646532, 720575940650935673,
            720575940653577974, 720575940653625846, 720575940661289345,
        ],
        'neu_exc2': [],
        'neu_slnc': [],
        'stim_rate': 200.0,
    },
}

DEFAULT_EXPERIMENT = 'sugar'


def get_experiment(name=None):
    """Return experiment config dict by name (default: sugar)."""
    name = name or DEFAULT_EXPERIMENT
    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{name}'. "
            f"Available: {list(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[name]

# ============================================================================
# Logging Utilities
# ============================================================================

class BenchmarkLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_file=None):
        self.log_file = log_file
        self.file_handle = None
        if log_file:
            self.file_handle = open(log_file, 'a', encoding='utf-8')

    def log(self, message, end='\n'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] {message}"
        print(formatted, end=end, flush=True)
        if self.file_handle:
            self.file_handle.write(formatted + end)
            self.file_handle.flush()

    def log_raw(self, message, end='\n'):
        """Log without timestamp."""
        print(message, end=end, flush=True)
        if self.file_handle:
            self.file_handle.write(message + end)
            self.file_handle.flush()

    def close(self):
        if self.file_handle:
            self.file_handle.close()

# ============================================================================
# CSV Result Persistence
# ============================================================================

CSV_COLUMNS = [
    'framework', 'n_run', 't_run',
    'setup_time', 'build_time', 'sim_time', 'total_time',
    'realtime_ratio', 'spikes', 'active_neurons', 'status', 'timestamp',
]


def save_result_csv(backend_name, result):
    """Append or update a benchmark result row in the CSV file.

    Uses (framework, n_run, t_run) as the composite key.  If a row with the
    same key already exists it is replaced; otherwise a new row is appended.
    """
    path_res.mkdir(parents=True, exist_ok=True)

    t = result.get('timings', {})

    row = {
        'framework': backend_name,
        'n_run': result['n_run'],
        't_run': result['t_run_sec'],
        'setup_time': round(t.get('network_creation_total',
                                  t.get('model_setup_total', 0)), 3),
        'build_time': round(t.get('device_build', 0), 3),
        'sim_time': round(t.get('simulation_total', 0), 3),
        'total_time': round(t.get('total_elapsed', 0), 3),
        'realtime_ratio': round(t.get('realtime_ratio', 0), 4),
        'spikes': result.get('n_spikes', 0),
        'active_neurons': result.get('n_active_neurons', 0),
        'status': result.get('status', 'unknown'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    key = (row['framework'], str(row['n_run']), str(row['t_run']))

    existing_rows = []
    if csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)

    updated = False
    for i, r in enumerate(existing_rows):
        existing_key = (r.get('framework', ''),
                        str(r.get('n_run', '')),
                        str(r.get('t_run', '')))
        if existing_key == key:
            existing_rows[i] = {k: str(v) for k, v in row.items()}
            updated = True
            break

    if not updated:
        existing_rows.append({k: str(v) for k, v in row.items()})

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(existing_rows)


# ============================================================================
# Summary Printing
# ============================================================================

def print_summary_table(all_results, backend_name, logger):
    """Print a formatted summary table for benchmark results."""
    logger.log_raw("")
    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"SUMMARY: {backend_name}")
    logger.log_raw("=" * 80)
    logger.log_raw("")
    logger.log_raw(
        f"{'t_run':>8} | {'n_run':>6} | {'Setup':>10} | "
        f"{'Build':>10} | {'Simulation':>12} | {'Total':>10} | "
        f"{'RT Ratio':>10} | {'Spikes':>10} | Status"
    )
    logger.log_raw("-" * 110)

    for result in all_results:
        t = result.get('timings', {})
        status_icon = "OK" if result['status'] == 'success' else "FAIL"

        setup_time = t.get(
            'network_creation_total', t.get('model_setup_total', 0)
        )
        build_time = t.get('device_build', 0)
        sim_time = t.get('simulation_total', 0)
        total_time = t.get('total_elapsed', 0)
        realtime_ratio = t.get('realtime_ratio', 0)

        logger.log_raw(
            f"{result['t_run_sec']:>7.1f}s | "
            f"{result['n_run']:>6d} | "
            f"{setup_time:>9.2f}s | "
            f"{build_time:>9.2f}s | "
            f"{sim_time:>11.2f}s | "
            f"{total_time:>9.2f}s | "
            f"{realtime_ratio:>9.3f}x | "
            f"{result['n_spikes']:>10d} | "
            f"{status_icon} {result['status']}"
        )

    logger.log_raw("-" * 110)
    logger.log_raw("")
    logger.log("Benchmark suite complete!")

# ============================================================================
# Backend Dispatcher
# ============================================================================

BACKEND_NAMES = {
    'cpu': 'Brian2 (CPU)',
    'gpu': 'Brian2CUDA (GPU)',
    'pytorch': 'PyTorch',
    'nestgpu': 'NEST GPU',
}


def run_benchmarks(backends, t_run_values=None, n_run_values=None,
                   experiment=None, logger=None):
    """
    Run benchmarks for the specified backends.

    Args:
        backends: list of backend keys ('cpu', 'gpu', 'pytorch', 'nestgpu')
        t_run_values: list of t_run durations in seconds, or None for all
        n_run_values: list of n_run values, or None for N_RUN_VALUES
        experiment: experiment config dict from get_experiment()
        logger: BenchmarkLogger instance

    Returns:
        dict mapping backend key to list of result dicts
    """
    if experiment is None:
        experiment = get_experiment()

    all_results = {}
    total_backends = len(backends)

    logger.log(f"Experiment: {experiment['name']}")
    logger.log(f"Stimulated neurons: {len(experiment['neu_exc'])} "
               f"at {experiment['stim_rate']} Hz")

    for bi, backend in enumerate(backends, 1):
        logger.log_raw("")
        logger.log(
            f">>> Starting backend {bi}/{total_backends}: "
            f"{BACKEND_NAMES[backend]}"
        )

        if backend in ('cpu', 'gpu'):
            from run_brian2_cuda import run_all_benchmarks as run_brian2
            results = run_brian2(
                use_cuda=(backend == 'gpu'),
                t_run_values=t_run_values,
                n_run_values=n_run_values,
                experiment=experiment,
                logger=logger,
            )
            all_results[backend] = results

        elif backend == 'pytorch':
            from run_pytorch import run_all_benchmarks as run_torch
            results = run_torch(
                t_run_values=t_run_values,
                n_run_values=n_run_values,
                experiment=experiment,
                logger=logger,
            )
            all_results[backend] = results

        elif backend == 'nestgpu':
            from run_nestgpu import run_all_benchmarks as run_nest
            results = run_nest(
                t_run_values=t_run_values,
                n_run_values=n_run_values,
                experiment=experiment,
                logger=logger,
            )
            all_results[backend] = results

        logger.log(
            f"<<< Finished backend {bi}/{total_backends}: "
            f"{BACKEND_NAMES[backend]}"
        )

    logger.log_raw("")
    logger.log(f"All {total_backends} backend(s) complete.")
    if csv_path.exists():
        logger.log(f"Results CSV: {csv_path}")

    return all_results

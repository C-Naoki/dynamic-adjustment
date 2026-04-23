import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.synthetics import ModelParams, generate_ground_truth, generate_params, load_data
from src.models.dynamicra.empirical import run_single as run_empirical_single
from src.models.dynamicra.model import DynamicRAEstimator, EstimateResult
from src.utils import build_arrays, configure_thread_limits, count_all_W_patterns, set_seed

RunIteration = Callable[[Any, DictConfig, Optional[tqdm], Optional[int]], List[EstimateResult]]


def simulate(cfg: DictConfig, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    if data is not None and cfg.verbose:
        print(count_all_W_patterns(data, cfg.data.T))
    run_iteration: RunIteration
    iteration_input: Any
    result_payload: Dict[str, Any]
    if data is not None:
        run_iteration = run_single
        iteration_input = data
        result_payload = {'raw_df': data}
        close_progress = True
    else:
        model_params = generate_params(cfg=cfg.data)
        paths = [tuple(cfg.model.treatment_path)]
        control_path = tuple(cfg.model.control_path)
        if control_path not in paths:
            paths.append(control_path)
        truth = {
            'mu_by_path': {
                path: generate_ground_truth(model_params, path, cfg.data, seed=cfg.exp.seed) for path in paths
            }
        }
        run_iteration = simulate_single
        iteration_input = model_params
        result_payload = {'model_params': model_params, 'truth': truth}
        close_progress = False

    if (cfg.exp.iterations <= 1) or (cfg.exp.n_workers <= 1):
        full_results = _run_iterations(cfg, run_iteration, iteration_input, close_progress=close_progress)
    else:
        full_results = _run_parallel_iterations(cfg, run_iteration, iteration_input)

    return {'full_results': full_results, 'config': cfg, **result_payload}


def _run_iterations(
    cfg: DictConfig,
    run_iteration: RunIteration,
    iteration_input: Any,
    close_progress: bool = False,
) -> List[List[EstimateResult]]:
    show_progress = bool(cfg.verbose)
    full_results = []
    bar_kwargs = {'dynamic_ncols': True, 'disable': not show_progress}
    outer = tqdm(range(cfg.exp.iterations), desc='Iterations', unit='iter', position=0, leave=True, **bar_kwargs)
    status = tqdm(total=1, position=1, leave=False, bar_format='{desc}', **bar_kwargs)

    for i in outer:
        results_per_time = run_iteration(iteration_input, cfg, status, cfg.exp.seed + 10007 * i)
        full_results.append(results_per_time)
        status.set_description_str('')
        status.refresh()

    if close_progress:
        status.close()
        outer.close()

    return full_results


def _run_parallel_iterations(
    cfg: DictConfig,
    run_iteration: RunIteration,
    iteration_input: Any,
) -> List[List[EstimateResult]]:
    cfg_container_obj = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_container_obj, dict):
        raise TypeError('OmegaConf.to_container(cfg, resolve=True) did not return a dict as expected.')
    cfg_container: Dict[str, Any] = cast(Dict[str, Any], cfg_container_obj)
    cfg_container['verbose'] = False  # prevent tqdm in workers

    results_buf: List[Optional[List[EstimateResult]]] = [None] * cfg.exp.iterations
    mp_ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=cfg.exp.n_workers,
        mp_context=mp_ctx,
        initializer=configure_thread_limits,
    ) as executor:
        futures = {
            executor.submit(
                _run_iteration_worker,
                cfg_container,
                run_iteration,
                iteration_input,
                cfg.exp.seed + 10007 * i,
            ): i
            for i in range(cfg.exp.iterations)
        }

        bar_kwargs = {'dynamic_ncols': True, 'disable': not bool(cfg.verbose)}
        outer = tqdm(
            as_completed(futures),
            total=cfg.exp.iterations,
            desc=f'Iterations (parallel, jobs={cfg.exp.n_workers})',
            unit='iter',
            position=0,
            leave=True,
            **bar_kwargs,
        )
        for fut in outer:
            i = futures[fut]
            results_buf[i] = fut.result()

    missing = next((i for i, res in enumerate(results_buf) if res is None), None)
    if missing is not None:
        raise RuntimeError(f'Parallel simulation did not produce a result for iteration {missing}.')

    return cast(List[List[EstimateResult]], results_buf)


def simulate_single(
    model_params: ModelParams,
    cfg: DictConfig,
    status: Optional[tqdm] = None,
    seed: Optional[int] = None,
) -> List[EstimateResult]:
    df, _ = load_data(
        cfg=cfg.data,
        model_params=model_params,
        seed=seed,
        verbose=False,
    )
    return run_single(data=df, cfg=cfg, status=status, seed=seed)


def _run_iteration_worker(
    cfg_container: Dict[str, Any],
    run_iteration: RunIteration,
    iteration_input: Any,
    seed: int,
) -> List[EstimateResult]:
    cfg = OmegaConf.create(cfg_container)
    cfg.verbose = False
    set_seed(seed, use_gpu=True)
    return run_iteration(iteration_input, cfg, None, seed)


def run_single(
    data: pd.DataFrame,
    cfg: DictConfig,
    status: Optional[tqdm] = None,
    seed: Optional[int] = None,
) -> List[EstimateResult]:
    if cfg.model.method == 'empirical':
        return run_empirical_single(data=data, cfg=cfg, status=status, seed=seed)

    X, Y, W, _ = build_arrays(data, cfg)
    treatment_path = np.asarray(cfg.model.treatment_path, dtype=int)
    control_path = np.asarray(cfg.model.control_path, dtype=int)

    show_progress = cfg.verbose
    model = DynamicRAEstimator(
        method=cfg.model.method,
        regmodel_name=cfg.model.regmodel,
        transmodel_name=cfg.model.transmodel,
        n_folds=cfg.model.n_folds,
        n_forward_mc=cfg.model.n_forward_mc,
        history_window=cfg.model.history_window,
        seed=seed,
        verbose=show_progress,
    )

    return model.estimate(
        X=X,
        Y=Y,
        W=W,
        times=None,
        treatment_path=treatment_path,
        control_path=control_path,
        alpha=1 - cfg.eval.ci_level,
        variance_type=cfg.eval.variance_type,
        band_type=cfg.eval.band_type,
        n_bootstrap=cfg.eval.n_boot,
        status_bar=status,
    )

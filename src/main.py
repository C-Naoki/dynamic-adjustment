import logging
import resource
import socket
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.models.dynamicra.simulate import simulate
from src.utils import configure_thread_limits, cprint, print_cfg, set_seed
from src.utils.io_helper import IOHelper
from src.utils.metrics import calculate_metrics
from src.utils.visualizer import plot_estimator_path_diagnostics, plot_metrics_summary

log = logging.getLogger(__name__)
warnings.simplefilter('ignore')


@hydra.main(version_base=None, config_path='config', config_name='settings')
def main(cfg: DictConfig) -> None:
    configure_thread_limits()
    print_cfg(
        obj={
            'Current time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Hostname': socket.gethostname(),
            'Model': cfg.model.name,
            'Method': cfg.model.method,
            'Input dir': f'{cfg.io.input_dir} ({cfg.data.get("model_type", "N/A")})',
            'Regression': cfg.model.get('regmodel', 'N/A'),
            'Transition': cfg.model.get('transmodel', 'N/A'),
        },
        title='Experimental Metadata',
        show_types=False,
        unicode_box=True,
    )
    set_seed(cfg.exp.seed, use_gpu=cfg.use_gpu)
    ioh = IOHelper(io_cfg=cfg.io)

    # create output directory path
    cfg.io.out_dir = ioh.create_path(cfg)

    # default treatment and control paths if not provided
    if cfg.model.get('treatment_path') is None:
        cfg.model.treatment_path = [1] * cfg.data.T
    if cfg.model.get('control_path') is None:
        cfg.model.control_path = [0] * cfg.data.T
    if len(cfg.model.treatment_path) != cfg.data.T or len(cfg.model.control_path) != cfg.data.T:
        raise ValueError('The length of treatment_path and control_path must match data.T.')

    st = time.monotonic()
    if cfg.io.input_dir == 'synthetics':
        results = simulate(cfg)
        truth = results['truth']
    else:
        df, metadata = ioh.load_data(cfg=cfg.data, seed=cfg.exp.seed, verbose=cfg.verbose)
        truth, covariate, outcome = metadata['truth'], metadata['covariate'], metadata['outcome']
        results = simulate(cfg, data=df)
        ioh.out_dir += f'covariate={covariate}/outcome={outcome}/'
    en = time.monotonic() - st
    print(f'>>>>> Elapsed time: {en:.3f}s <<<<<')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2) if sys.platform == 'darwin' else peak / 1024
    print(f'>>>>> Peak memory usage: {peak_mib:.3f} MiB <<<<<')

    # additional info to results
    metrics = calculate_metrics(results['full_results'], truth, cfg)
    results['truth'] = truth
    results['metrics'] = metrics
    print(metrics['overall'])

    # create output directory
    if cfg.save:
        ioh.mkdir()
        for key, value in results.items():
            ioh.savepkl(obj=value, name=key)
        out_dir = Path(ioh.out_dir)
        fig_path = out_dir / 'metrics_summary.png' if cfg.save else None
        diag_path = out_dir / 'path_diagnostics.png' if cfg.save else None
        if truth is not None:
            plot_metrics_summary(metrics, model_name=cfg.model.method, save_path=fig_path)
        plot_estimator_path_diagnostics(metrics, save_path=diag_path)

    cprint('Done!')


if __name__ == '__main__':
    main()

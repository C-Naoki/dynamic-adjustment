import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

line_width = 2.5
styles_kwargs = {
    'font.family': 'Times New Roman',
    'axes.edgecolor': 'black',
}
legend_kwargs = {
    'frameon': True,
    'handlelength': 1.2,
}

def plot_metrics_summary(
    metrics: Dict[str, Any],
    figsize: Tuple[float, float] = (11.0, 7.0),
    style: str = 'whitegrid',
    model_name: str = 'DynamicRA',
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: Optional[int] = None,
) -> plt.Figure:
    if not metrics:
        raise ValueError('metrics must be a non-empty dictionary.')
    per_time = metrics['per_time']
    if not per_time:
        raise ValueError('metrics does not contain "per_time" entries to plot.')

    # Extract series needed for plotting
    times = tuple(int(entry['t']) for entry in per_time)
    truth = _ensure_float_sequence((entry['truth'] for entry in per_time), key='truth')
    mean_hat = _ensure_float_sequence((entry['mean_hat'] for entry in per_time), key='mean_hat')
    ci_low = []
    ci_high = []
    for entry in per_time:
        lo = entry['bootstrap_mean_ci_low']
        hi = entry['bootstrap_mean_ci_high']
        if lo is None or hi is None:
            raise ValueError('Confidence interval bounds are missing for at least one horizon.')
        ci_low.append(lo)
        ci_high.append(hi)
    bias = _ensure_float_sequence((entry['bias'] for entry in per_time), key='bias')
    rmse = _ensure_float_sequence((entry['rmse'] for entry in per_time), key='rmse')
    coverage = _ensure_float_sequence((entry['bootstrap_coverage'] for entry in per_time), key='bootstrap_coverage')

    iterations = metrics['iterations']
    ci_level = metrics['ci_level']
    treat_path = metrics['treatment_path']
    control_path = metrics.get('control_path')
    estimand = metrics.get('estimand', 'path_mean')
    if estimand == 'ate' and treat_path is not None and control_path is not None:
        path_label = f'ATE: {tuple(treat_path)} - {tuple(control_path)}'
    else:
        path_label = f'path={tuple(treat_path)}' if treat_path is not None else 'path=?'
    if model_name == 'dynamicra':
        model_name = 'DynamicRA'
    elif model_name == 'empirical':
        model_name = 'Empirical'

    with sns.axes_style(style):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={'height_ratios': (2.0, 1.0)},
        )
    if dpi is not None:
        fig.set_dpi(dpi)

    ax_main, ax_err = axes

    # Panel 1: truth vs estimate with confidence band
    ax_main.plot(times, truth, label='Truth', color='tab:blue', linewidth=2.0)
    ax_main.plot(times, mean_hat, label='Mean estimate', color='tab:orange', marker='o')
    ax_main.fill_between(
        times,
        tuple(ci_low),
        tuple(ci_high),
        color='tab:orange',
        alpha=0.2,
        label=f'{int(ci_level * 100)}% CI',
    )
    ax_main.set_ylabel('Outcome')
    ax_main.set_title(
        f'{model_name} metrics ({iterations} iterations, {path_label})',
        fontsize='medium',
    )
    ax_main.legend(loc='best')

    # Panel 2: Bias & RMSE along with coverage on secondary axis
    ax_err.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    ax_err.plot(times, bias, label='Bias', color='tab:red', marker='o')
    ax_err.plot(times, rmse, label='RMSE', color='tab:purple', marker='s')
    ax_err.set_ylabel('Bias / RMSE')

    ax_cov = ax_err.twinx()
    ax_cov.plot(times, coverage, label='Coverage', color='tab:green', marker='^', linestyle='--')
    ax_cov.axhline(ci_level, color='tab:green', linestyle=':', linewidth=1.0, alpha=0.8)
    ax_cov.set_ylabel('Coverage')
    ax_cov.set_ylim(0.0, 1.05)

    # Combine legends for second panel
    handles_err, labels_err = ax_err.get_legend_handles_labels()
    handles_cov, labels_cov = ax_cov.get_legend_handles_labels()
    ax_err.legend(handles_err + handles_cov, labels_err + labels_cov, loc='upper right')

    ax_err.set_xlabel('Time horizon (t)')
    ax_err.set_xticks(times)

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_time_horizon_metrics(
    metrics_by_label: Mapping[str, Dict[str, Any]],
    figsize: Tuple[float, float] = (14.0, 4.5),
    w_pad: float = 3.0,
    style: str = 'ticks',
    markers: Optional[Sequence[str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    xlabel: str = 'Time horizon',
    ylabel: str = 'Value',
    legend_loc: str = 'upper center',
    legend_bbox_to_anchor: Tuple[float, float] = (0.5, -0.25),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 100,
) -> plt.Figure:
    if not metrics_by_label:
        raise ValueError('metrics_by_label must be a non-empty mapping.')

    labels = tuple(metrics_by_label.keys())
    marker_cycle = tuple(markers) if markers is not None else ('o', 's', 'D', '^', 'v', 'P', 'X', '*')
    if not marker_cycle:
        raise ValueError('markers must contain at least one marker symbol.')
    marker_lookup = {label: marker_cycle[idx % len(marker_cycle)] for idx, label in enumerate(labels)}

    first_times: Optional[Tuple[int, ...]] = None
    reference_ci_level: Optional[float] = None
    per_metric_series: Dict[str, Dict[str, Tuple[float, ...]]] = {
        'rmse': {},
        'bias': {},
        'variance': {},
        'ci_length': {},
        'coverage': {},
    }

    for label, metrics in metrics_by_label.items():
        if not metrics:
            raise ValueError(f'metrics for label "{label}" must be a non-empty dictionary.')
        per_time = metrics.get('per_time')
        if not per_time:
            raise ValueError(f'metrics for label "{label}" does not contain "per_time" entries.')

        times = tuple(int(entry['t']) + 1 for entry in per_time)
        if first_times is None:
            first_times = times
        elif times != first_times:
            raise ValueError(f'Horizon mismatch for label "{label}": {times} vs {first_times}.')

        rmse_values = []
        bias_values = []
        variance_values = []
        ci_length_values = []
        coverage_values = []
        for entry in per_time:
            mean_hat = _safe_float(entry.get('mean_hat'))
            truth = _safe_float(entry.get('truth'))

            bias = _safe_float(entry.get('bias'))
            if not math.isfinite(bias) and math.isfinite(mean_hat) and math.isfinite(truth):
                bias = mean_hat - truth

            std_hat = _safe_float(entry.get('std_hat'))
            if not math.isfinite(std_hat):
                std_hat = _safe_float(entry.get('bootstrap_mean_se'))
            variance = std_hat * std_hat if math.isfinite(std_hat) else math.nan

            rmse = _safe_float(entry.get('rmse'))
            if not math.isfinite(rmse) and math.isfinite(variance) and math.isfinite(bias):
                rmse = math.sqrt(variance + bias * bias)

            ci_low = _safe_float(entry.get('bootstrap_mean_ci_low'))
            ci_high = _safe_float(entry.get('bootstrap_mean_ci_high'))
            if not math.isfinite(ci_low) or not math.isfinite(ci_high):
                ci_low = _safe_float(entry.get('empirical_ci_low'))
                ci_high = _safe_float(entry.get('empirical_ci_high'))
            if math.isfinite(ci_low) and math.isfinite(ci_high):
                ci_length_values.append(ci_high - ci_low)
            else:
                raise ValueError(
                    'Unable to derive CI length for one or more horizons. '
                    'Ensure metrics were computed with an updated calculate_metrics implementation.',
                )

            coverage = _safe_float(entry.get('bootstrap_coverage'))
            if (
                not math.isfinite(coverage)
                and math.isfinite(ci_low)
                and math.isfinite(ci_high)
                and math.isfinite(truth)
            ):
                coverage = float(ci_low <= truth <= ci_high)

            rmse_values.append(rmse)
            bias_values.append(bias)
            variance_values.append(variance)
            coverage_values.append(coverage)

        rmse_series = tuple(rmse_values)
        bias_series = tuple(bias_values)
        variance_series = tuple(variance_values)
        ci_length_series = tuple(ci_length_values)
        coverage_series = tuple(coverage_values)

        per_metric_series['rmse'][label] = rmse_series
        per_metric_series['bias'][label] = bias_series
        per_metric_series['variance'][label] = variance_series
        per_metric_series['ci_length'][label] = ci_length_series
        per_metric_series['coverage'][label] = coverage_series

        ci_level = metrics.get('ci_level')
        if ci_level is not None:
            if reference_ci_level is None:
                reference_ci_level = float(ci_level)
            elif not math.isclose(float(ci_level), reference_ci_level, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f'All metrics must share the same ci_level. "{label}" had {ci_level}, '
                    f'expected {reference_ci_level}.',
                )

    assert first_times is not None  # for type-checkers
    times = first_times

    default_colors = sns.color_palette('tab10', n_colors=len(labels))
    color_lookup = {
        label: colors[label] if colors is not None and label in colors else default_colors[idx % len(default_colors)]
        for idx, label in enumerate(labels)
    }

    metric_layout = (
        ('rmse', 'RMSE'),
        # ('bias', 'Bias'),
        # ('variance', 'Variance'),
        ('ci_length', 'Average CI Length'),
        ('coverage', 'Coverage Probability'),
    )

    with sns.axes_style(style, rc=styles_kwargs):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=figsize,
            sharex=True,
        )
        axes_arr = np.atleast_1d(axes).ravel()
        sns.despine(fig=fig)

        for ax in axes_arr:
            ax.spines['left'].set_linewidth(line_width)
            ax.spines['bottom'].set_linewidth(line_width)
            ax.tick_params(width=line_width)

        fig.set_dpi(dpi)

    axes = np.atleast_1d(axes).ravel()
    plot_axes = axes[: len(metric_layout)]
    for extra_ax in axes[len(metric_layout) :]:
        extra_ax.set_visible(False)

    for axis_idx, (metric_key, title) in enumerate(metric_layout):
        ax = plot_axes[axis_idx]
        for label in labels:
            series = per_metric_series[metric_key][label]
            marker = marker_lookup[label]
            ax.plot(
                times,
                series,
                label=label,
                color=color_lookup[label],
                marker=marker,
                markerfacecolor='none',
                markersize=12,
                markeredgewidth=3,
                linewidth=2.0,
            )
        ax.set_title(title, fontsize=32)
        ax.set_xlabel(xlabel, fontsize=32)
        if axis_idx == 0:
            ax.set_ylabel(ylabel, fontsize=32)
            ax.set_ylim(bottom=0.0)
            if 0.14 < max(series) < 0.2:
                ax.set_yticks([0.0, 0.07, 0.14])
        ax.set_xticks(times)
        ax.tick_params(labelsize=27, labelleft=True)
        if axis_idx == 1:
            ax.set_ylim(bottom=0.0)
            if 0.8 < max(series):
                ax.set_yticks([0.0, 0.4, 0.8])
            elif 0.5 < max(series) <= 0.8:
                ax.set_yticks([0.0, 0.3, 0.6])
        if axis_idx == 2:
            ax.axhline(reference_ci_level, color='black', linestyle=':', linewidth=3.0, zorder=0)
            # ax.set_yticks([0.94, 0.95, 0.96])
            # ax.set_ylim(0.938, 0.968)
        ax.grid(False)

    # Adjust coverage axis specifics
    coverage_idx = next(idx for idx, (metric_key, _) in enumerate(metric_layout) if metric_key == 'coverage')
    cov_ax = plot_axes[coverage_idx]
    coverage_values = [
        value for series in per_metric_series['coverage'].values() for value in series if math.isfinite(value)
    ]
    if reference_ci_level is not None:
        coverage_values.append(reference_ci_level)

    if coverage_values:
        coverage_min = min(coverage_values)
        coverage_max = max(coverage_values)
        span = coverage_max - coverage_min
        min_span = 0.05
        if span < min_span:
            center = 0.5 * (coverage_max + coverage_min)
            lower = center - min_span / 2.0
            upper = center + min_span / 2.0
        else:
            padding = span * 0.1
            lower = coverage_min - padding
            upper = coverage_max + padding
        lower = max(0.0, lower)
        if upper - lower < min_span:
            upper = lower + min_span
        # cov_ax.set_ylim(lower, upper)
    else:
        cov_ax.set_ylim(0.0, 1.05)

    # if reference_ci_level is not None:
        # cov_ax.axhline(reference_ci_level, color='red', linestyle='--', linewidth=2.5, alpha=0.8, zorder=0)

    # Construct shared legend beneath the panels
    legend_handles, legend_labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=min(len(labels), 5),
        fontsize=27,
        columnspacing=0.9,
        handletextpad=0.4,
        **legend_kwargs,
    )

    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0), w_pad=w_pad)
    fig.subplots_adjust(bottom=max(0.2, -legend_bbox_to_anchor[1]))

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    return fig


def _ensure_float_sequence(values: Iterable[Any], key: str) -> Tuple[float, ...]:
    try:
        return tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError(f'Unable to convert values for "{key}" to float.') from exc


def _safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def plot_variance_ratio(
    metrics_numerator: Dict[str, Any],
    metrics_denominator: Dict[str, Any],
    numerator_label: str = 'Model 1',
    denominator_label: str = 'Model 2',
    figsize: Tuple[float, float] = (10.0, 6.0),
    style: str = 'whitegrid',
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: Optional[int] = None,
) -> plt.Figure:
    if not metrics_numerator:
        raise ValueError('metrics_numerator must be a non-empty dictionary.')
    if not metrics_denominator:
        raise ValueError('metrics_denominator must be a non-empty dictionary.')

    per_time_num = metrics_numerator.get('per_time')
    if not per_time_num:
        raise ValueError('metrics_numerator does not contain "per_time" entries to plot.')
    per_time_den = metrics_denominator.get('per_time')
    if not per_time_den:
        raise ValueError('metrics_denominator does not contain "per_time" entries to plot.')

    times_num = tuple(int(entry['t']) for entry in per_time_num)
    times_den = tuple(int(entry['t']) for entry in per_time_den)
    if times_den != times_num:
        raise ValueError(f'Horizon mismatch: numerator {times_num} vs denominator {times_den}.')

    stds_num = _ensure_float_sequence((entry['std_hat'] for entry in per_time_num), key='std_hat')
    stds_den = _ensure_float_sequence((entry['std_hat'] for entry in per_time_den), key='std_hat')

    variances_num = tuple(val * val for val in stds_num)
    variances_den = tuple(val * val for val in stds_den)
    variance_ratio = tuple(
        (num / den) if math.isfinite(num) and math.isfinite(den) and den not in (0.0, -0.0) else math.nan
        for num, den in zip(variances_num, variances_den)
    )

    with sns.axes_style(style):
        fig, ax = plt.subplots(figsize=figsize)
    if dpi is not None:
        fig.set_dpi(dpi)

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', label='Parity')
    ax.plot(
        times_num,
        variance_ratio,
        marker='o',
        linewidth=2.0,
        color='tab:blue',
        label=f'Var({numerator_label}) / Var({denominator_label})',
    )
    ax.set_xlabel('Time horizon')
    ax.set_ylabel('Variance ratio')
    ax.set_title(
        f'Variance ratio between {numerator_label} and {denominator_label}',
    )
    ax.set_xticks(times_num)
    ax.legend(loc='best')

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_rmse_reduction(
    metrics_ls: Sequence[Sequence[Dict[str, Any]]],
    baseline_metrics_ls: Sequence[Sequence[Dict[str, Any]]],
    baseline_label: str = 'Baseline',
    model_label: str = 'DynamicRA',
    curve_labels: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
    w_pad: float = 3.0,
    style: str = 'whitegrid',
    title: Optional[str] = None,
    subtitles: Optional[Sequence[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 300,
) -> plt.Figure:

    def _materialise_panels(
        value: Any,
        name: str,
    ) -> Tuple[Tuple[Dict[str, Any], ...], ...]:
        if isinstance(value, dict):
            return ((value,),)

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                raise ValueError(f'{name} must be a non-empty sequence.')

            # Legacy: sequence of dictionaries -> single panel.
            if all(isinstance(entry, dict) for entry in value):
                return (tuple(value),)

            panels = []
            for panel_idx, panel in enumerate(value):
                if not isinstance(panel, Sequence) or isinstance(panel, (str, bytes)):
                    raise TypeError(f'{name}[{panel_idx}] must be a sequence of dictionaries.')
                if not panel:
                    raise ValueError(f'{name}[{panel_idx}] must be a non-empty sequence.')
                if not all(isinstance(entry, dict) for entry in panel):
                    raise TypeError(f'All entries in {name}[{panel_idx}] must be dictionaries.')
                panels.append(tuple(panel))
            return tuple(panels)

        raise TypeError(f'{name} must be a sequence of sequences of dictionaries.')

    metrics_panels = _materialise_panels(metrics_ls, 'metrics_ls')
    baseline_panels = _materialise_panels(baseline_metrics_ls, 'baseline_metrics_ls')

    if len(metrics_panels) != len(baseline_panels):
        raise ValueError('metrics_ls and baseline_metrics_ls must have the same number of panels.')

    curve_labels_tuple = tuple(curve_labels) if curve_labels is not None else None

    for panel_idx, (metrics_panel, baseline_panel) in enumerate(zip(metrics_panels, baseline_panels)):
        if len(metrics_panel) != len(baseline_panel):
            raise ValueError(
                'metrics_ls and baseline_metrics_ls must have the same number of entries within each panel. '
                f'Panel {panel_idx} had {len(metrics_panel)} vs {len(baseline_panel)}.',
            )
        if curve_labels_tuple is not None and len(curve_labels_tuple) != len(metrics_panel):
            raise ValueError(
                'curve_labels length must match the number of curves in every panel. '
                f'Expected {len(metrics_panel)} labels for panel {panel_idx} but got {len(curve_labels_tuple)}.',
            )

    def _legend_labels(num_curves: int) -> Tuple[str, ...]:
        if curve_labels_tuple is not None:
            return curve_labels_tuple
        if num_curves == 1:
            return (f'{model_label} vs {baseline_label}',)
        return tuple(f'{model_label} vs {baseline_label} #{idx}' for idx in range(1, num_curves + 1))

    marker_cycle = ('o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8')

    panel_times = []
    panel_reduction_series = []
    panel_legend_labels = []
    for panel_idx, (metrics_panel, baseline_panel) in enumerate(zip(metrics_panels, baseline_panels)):
        reference_times: Optional[Tuple[int, ...]] = None
        computed_reductions = []
        for curve_idx, (metrics, baseline_metrics) in enumerate(zip(metrics_panel, baseline_panel)):
            if not metrics:
                raise ValueError(f'metrics_ls[{panel_idx}][{curve_idx}] must be a non-empty dictionary.')
            per_time = metrics.get('per_time')
            if not per_time:
                raise ValueError(
                    f'metrics_ls[{panel_idx}][{curve_idx}] does not contain "per_time" entries to plot.',
                )

            times = tuple(int(entry['t']) + 1 for entry in per_time)
            if reference_times is None:
                reference_times = times
            elif times != reference_times:
                raise ValueError(
                    f'Horizons {times} in metrics_ls[{panel_idx}][{curve_idx}] do not match {reference_times}.',
                )

            model_rmse = _ensure_float_sequence(
                (entry['rmse'] for entry in per_time),
                key=f'metrics_ls[{panel_idx}][{curve_idx}].rmse',
            )

            if not baseline_metrics:
                raise ValueError(
                    f'baseline_metrics_ls[{panel_idx}][{curve_idx}] must be a non-empty dictionary.',
                )
            baseline_per_time = baseline_metrics.get('per_time')
            if not baseline_per_time:
                raise ValueError(
                    f'baseline_metrics_ls[{panel_idx}][{curve_idx}] must contain non-empty "per_time" entries.',
                )
            baseline_times = times
            if baseline_times != reference_times:
                raise ValueError(
                    'Baseline horizons '
                    f'{baseline_times} in baseline_metrics_ls[{panel_idx}][{curve_idx}] do not match {reference_times}.',
                )
            baseline_rmse_series = _ensure_float_sequence(
                (entry['rmse'] for entry in baseline_per_time),
                key=f'baseline_metrics_ls[{panel_idx}][{curve_idx}].rmse',
            )

            computed_reductions.append(
                tuple(
                    (
                        (base - model) / base * 100.0
                        if math.isfinite(base) and base not in (0.0, -0.0) and math.isfinite(model)
                        else math.nan
                    )
                    for base, model in zip(baseline_rmse_series, model_rmse)
                ),
            )

        if reference_times is None:  # pragma: no cover - defensive guard
            raise ValueError(f'metrics_ls[{panel_idx}] did not contain any plottable entries.')

        panel_times.append(reference_times)
        panel_reduction_series.append(tuple(computed_reductions))
        panel_legend_labels.append(_legend_labels(len(computed_reductions)))

    n_panels = len(panel_times)
    total_figsize = figsize if n_panels == 1 else (figsize[0] * n_panels, figsize[1])
    use_shared_legend = n_panels > 1 and len({len(series) for series in panel_reduction_series}) == 1

    with sns.axes_style(style, rc=styles_kwargs):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_panels,
            figsize=total_figsize,
            sharey=n_panels > 1,
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(axes).ravel()
        sns.despine(fig=fig)

        for ax in axes_arr:
            ax.spines['left'].set_linewidth(line_width)
            ax.spines['bottom'].set_linewidth(line_width)
            ax.tick_params(width=line_width)
            ax.set_yticks([0, 25, 50, 75])
            ax.set_ylim(0, 75)

        fig.set_dpi(dpi)

    if title is not None and n_panels > 1:
        fig.suptitle(title, fontsize='medium')

    for panel_idx, ax in enumerate(axes_arr[:n_panels]):
        times = panel_times[panel_idx]
        reduction_series = panel_reduction_series[panel_idx]
        legend_labels = panel_legend_labels[panel_idx]

        for curve_idx, reduction_pct in enumerate(reduction_series):
            marker = marker_cycle[curve_idx % len(marker_cycle)]
            ax.plot(
                times,
                reduction_pct,
                marker=marker,
                markerfacecolor='none',
                markersize=12,
                markeredgewidth=3,
                linewidth=2.0,
                label=legend_labels[curve_idx],
            )
            ax.tick_params(labelsize=27, labelleft=True)

        ax.set_xlabel('Time horizon', fontsize=32)
        if panel_idx == 0:
            ax.set_ylabel('RMSE Reduction', fontsize=31)
        ax.set_xticks(times)
        # ax.set_yticks([0, 25, 50])

        if n_panels == 1:
            if title is not None:
                ax.set_title(title, fontsize=32)
            elif len(reduction_series) == 1:
                ax.set_title(f'RMSE reduction of {model_label} relative to {baseline_label}', fontsize=32)
            else:
                ax.set_title('RMSE reduction across estimator comparisons')
            ax.legend(loc='best')
        else:
            if subtitles is not None and panel_idx < len(subtitles):
                ax.set_title(subtitles[panel_idx], fontsize=32)
            else:
                ax.set_title(f'Panel {panel_idx + 1}')
            if not use_shared_legend:
                ax.legend(loc='best')

    # fig.supxlabel('Time horizon', fontsize=32, y=0.16)
    if n_panels == 1:
        fig.tight_layout()
    else:
        top = 0.9 if title is not None else 1.0
        if use_shared_legend:
            legend_handles, legend_labels = axes_arr[0].get_legend_handles_labels()
            if legend_handles:
                fig.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.08),
                    ncol=len(panel_legend_labels[0]),
                    fontsize=27,
                    **legend_kwargs,
                )
            fig.tight_layout(rect=(0.0, 0.12, 1.0, top), w_pad=w_pad)
        else:
            fig.tight_layout(rect=(0.0, 0.0, 1.0, top), w_pad=w_pad)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_bias_reduction(
    metrics_ls: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    baseline_metrics_ls: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    baseline_label: str = 'Baseline',
    model_label: str = 'DynamicRA',
    curve_labels: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
    style: str = 'whitegrid',
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 300,
) -> plt.Figure:

    def _materialise_metrics(
        value: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        name: str,
    ) -> Tuple[Dict[str, Any], ...]:
        if isinstance(value, dict):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                raise ValueError(f'{name} must be a non-empty sequence.')
            if not all(isinstance(entry, dict) for entry in value):
                raise TypeError(f'All entries in {name} must be dictionaries.')
            return tuple(value)
        raise TypeError(f'{name} must be a metrics dictionary or a sequence of dictionaries.')

    metrics_seq = _materialise_metrics(metrics_ls, 'metrics_ls')
    baseline_seq = _materialise_metrics(baseline_metrics_ls, 'baseline_metrics_ls')

    if len(metrics_seq) != len(baseline_seq):
        raise ValueError('metrics_ls and baseline_metrics_ls must have the same length.')

    if curve_labels is not None:
        curve_labels = tuple(curve_labels)
        if len(curve_labels) != len(metrics_seq):
            raise ValueError('curve_labels length must match the number of metrics entries.')
    else:
        curve_labels = None

    reference_times: Optional[Tuple[int, ...]] = None
    reduction_series: Tuple[Tuple[float, ...], ...]
    computed_reductions = []
    for idx, (metrics, baseline_metrics) in enumerate(zip(metrics_seq, baseline_seq), start=1):
        if not metrics:
            raise ValueError(f'metrics_ls[{idx - 1}] must be a non-empty dictionary.')
        per_time = metrics.get('per_time')
        if not per_time:
            raise ValueError(f'metrics_ls[{idx - 1}] does not contain "per_time" entries to plot.')

        times = tuple(int(entry['t']) for entry in per_time)
        if reference_times is None:
            reference_times = times
        elif times != reference_times:
            raise ValueError(
                f'Horizons {times} in metrics_ls[{idx - 1}] do not match {reference_times}.',
            )

        model_bias = _ensure_float_sequence((entry['bias'] for entry in per_time), key='bias')
        model_abs_bias = tuple(abs(val) for val in model_bias)

        if not baseline_metrics:
            raise ValueError(f'baseline_metrics_ls[{idx - 1}] must be a non-empty dictionary.')
        baseline_per_time = baseline_metrics.get('per_time')
        if not baseline_per_time:
            raise ValueError(
                f'baseline_metrics_ls[{idx - 1}] must contain non-empty "per_time" entries.',
            )
        baseline_times = tuple(int(entry['t']) for entry in baseline_per_time)
        if baseline_times != reference_times:
            raise ValueError(
                'Baseline horizons '
                f'{baseline_times} in baseline_metrics_ls[{idx - 1}] do not match {reference_times}.',
            )
        baseline_bias_series = _ensure_float_sequence(
            (entry['bias'] for entry in baseline_per_time),
            key=f'baseline[{idx - 1}].bias',
        )
        baseline_abs_bias = tuple(abs(val) for val in baseline_bias_series)

        computed_reductions.append(
            tuple(
                ((base - model) / base * 100.0 if math.isfinite(base) and base not in (0.0, -0.0) else math.nan)
                for base, model in zip(baseline_abs_bias, model_abs_bias)
            ),
        )

    assert reference_times is not None  # for type checking
    reduction_series = tuple(computed_reductions)

    if curve_labels is not None:
        legend_labels = curve_labels
    elif len(reduction_series) == 1:
        legend_labels = (f'{model_label} vs {baseline_label}',)
    else:
        legend_labels = tuple(
            f'{model_label} vs {baseline_label} #{idx}' for idx in range(1, len(reduction_series) + 1)
        )

    marker_cycle = ('o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8')

    with sns.axes_style(style):
        fig, ax = plt.subplots(figsize=figsize)
    fig.set_dpi(dpi)

    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    for idx, reduction_pct in enumerate(reduction_series):
        marker = marker_cycle[idx % len(marker_cycle)]
        ax.plot(
            reference_times,
            reduction_pct,
            marker=marker,
            linewidth=2.0,
            label=legend_labels[idx],
        )
    ax.set_xlabel('Time horizon')
    ax.set_ylabel('Absolute Bias Reduction (%)')
    if title is not None:
        ax.set_title(title)
    elif len(reduction_series) == 1:
        ax.set_title(f'Absolute bias reduction of {model_label} relative to {baseline_label}')
    else:
        ax.set_title('Absolute bias reduction across estimator comparisons')
    ax.set_xticks(reference_times)
    ax.legend(loc='best')

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_variance_reduction(
    metrics_ls: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    baseline_metrics_ls: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    baseline_label: str = 'Baseline',
    model_label: str = 'DynamicRA',
    curve_labels: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
    style: str = 'whitegrid',
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 300,
) -> plt.Figure:

    def _materialise_metrics(
        value: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        name: str,
    ) -> Tuple[Dict[str, Any], ...]:
        if isinstance(value, dict):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                raise ValueError(f'{name} must be a non-empty sequence.')
            if not all(isinstance(entry, dict) for entry in value):
                raise TypeError(f'All entries in {name} must be dictionaries.')
            return tuple(value)
        raise TypeError(f'{name} must be a metrics dictionary or a sequence of dictionaries.')

    metrics_seq = _materialise_metrics(metrics_ls, 'metrics_ls')
    baseline_seq = _materialise_metrics(baseline_metrics_ls, 'baseline_metrics_ls')

    if len(metrics_seq) != len(baseline_seq):
        raise ValueError('metrics_ls and baseline_metrics_ls must have the same length.')

    if curve_labels is not None:
        curve_labels = tuple(curve_labels)
        if len(curve_labels) != len(metrics_seq):
            raise ValueError('curve_labels length must match the number of metrics entries.')
    else:
        curve_labels = None

    reference_times: Optional[Tuple[int, ...]] = None
    reduction_series: Tuple[Tuple[float, ...], ...]
    computed_reductions = []
    for idx, (metrics, baseline_metrics) in enumerate(zip(metrics_seq, baseline_seq), start=1):
        if not metrics:
            raise ValueError(f'metrics_ls[{idx - 1}] must be a non-empty dictionary.')
        per_time = metrics.get('per_time')
        if not per_time:
            raise ValueError(f'metrics_ls[{idx - 1}] does not contain "per_time" entries to plot.')

        times = tuple(int(entry['t']) for entry in per_time)
        if reference_times is None:
            reference_times = times
        elif times != reference_times:
            raise ValueError(
                f'Horizons {times} in metrics_ls[{idx - 1}] do not match {reference_times}.',
            )

        model_stds = _ensure_float_sequence((entry['std_hat'] for entry in per_time), key='std_hat')
        model_vars = tuple(val * val for val in model_stds)

        if not baseline_metrics:
            raise ValueError(f'baseline_metrics_ls[{idx - 1}] must be a non-empty dictionary.')
        baseline_per_time = baseline_metrics.get('per_time')
        if not baseline_per_time:
            raise ValueError(
                f'baseline_metrics_ls[{idx - 1}] must contain non-empty "per_time" entries.',
            )
        baseline_times = tuple(int(entry['t']) for entry in baseline_per_time)
        if baseline_times != reference_times:
            raise ValueError(
                'Baseline horizons '
                f'{baseline_times} in baseline_metrics_ls[{idx - 1}] do not match {reference_times}.',
            )
        baseline_stds = _ensure_float_sequence(
            (entry['std_hat'] for entry in baseline_per_time),
            key=f'baseline[{idx - 1}].std_hat',
        )
        baseline_vars = tuple(val * val for val in baseline_stds)

        computed_reductions.append(
            tuple(
                ((base - model) / base * 100.0 if math.isfinite(base) and base not in (0.0, -0.0) else math.nan)
                for base, model in zip(baseline_vars, model_vars)
            ),
        )

    assert reference_times is not None  # for type checking
    reduction_series = tuple(computed_reductions)

    if curve_labels is not None:
        legend_labels = curve_labels
    elif len(reduction_series) == 1:
        legend_labels = (f'{model_label} vs {baseline_label}',)
    else:
        legend_labels = tuple(
            f'{model_label} vs {baseline_label} #{idx}' for idx in range(1, len(reduction_series) + 1)
        )

    marker_cycle = ('o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8')

    with sns.axes_style(style):
        fig, ax = plt.subplots(figsize=figsize)
    fig.set_dpi(dpi)

    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    for idx, reduction_pct in enumerate(reduction_series):
        marker = marker_cycle[idx % len(marker_cycle)]
        ax.plot(
            reference_times,
            reduction_pct,
            marker=marker,
            linewidth=2.0,
            label=legend_labels[idx],
        )
    ax.set_xlabel('Time horizon')
    ax.set_ylabel('Variance Reduction (%)')
    if title is not None:
        ax.set_title(title)
    elif len(reduction_series) == 1:
        ax.set_title(f'Variance reduction of {model_label} relative to {baseline_label}')
    else:
        ax.set_title('Variance reduction across estimator comparisons')
    ax.set_xticks(reference_times)
    ax.legend(loc='best')

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_estimator_path_diagnostics(
    metrics: Dict[str, Any],
    figsize: Tuple[float, float] = (12.0, 8.0),
    style: str = 'whitegrid',
    dynamic_label: str = 'DynamicRA estimate',
    baseline_label: str = 'Baseline $G^{(1)}$',
    truth_label: str = 'Ground truth',
    comparison_series: Optional[Dict[str, Sequence[Any]]] = None,
    events: Optional[Iterable[Tuple[int, str]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: Optional[int] = None,
) -> plt.Figure:
    if not metrics:
        raise ValueError('metrics must be a non-empty dictionary.')
    per_time = metrics['per_time']
    if not per_time:
        raise ValueError('metrics does not contain "per_time" entries to plot.')

    times = tuple(int(entry['t']) for entry in per_time)
    mu_vals = _ensure_float_sequence((entry['mean_hat'] for entry in per_time), key='mean_hat')

    truth_vals = tuple(float(entry.get('truth', math.nan)) for entry in per_time)
    bias_vals = tuple(float(entry.get('bias', math.nan)) for entry in per_time)
    rmse_vals = tuple(float(entry.get('rmse', math.nan)) for entry in per_time)

    baseline_vals = tuple(entry['baseline_mean'] for entry in per_time)
    residual_vals = tuple(entry['residual_mean'] for entry in per_time)
    aug_vals = tuple(entry['trans_aug_mean'] for entry in per_time)

    ci_low_ls, ci_high_ls = [], []
    for entry in per_time:
        lo = entry['bootstrap_mean_ci_low']
        hi = entry['bootstrap_mean_ci_high']
        if lo is None or hi is None:
            lo = entry['empirical_ci_low']
            hi = entry['empirical_ci_high']
        ci_low_ls.append(lo if lo is not None else math.nan)
        ci_high_ls.append(hi if hi is not None else math.nan)
    ci_low = tuple(ci_low_ls)
    ci_high = tuple(ci_high_ls)

    iterations = metrics['iterations']
    path_label = metrics['treatment_path']

    with sns.axes_style(style):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={'height_ratios': (2.4, 1.4)},
        )
    if dpi is not None:
        fig.set_dpi(dpi)

    ax_main, ax_contrib = axes

    # Panel 1: DynamicRA trajectory vs. baseline (if present)
    ax_main.plot(times, mu_vals, label=dynamic_label, color='tab:blue', linewidth=2.2, marker='o')

    if not all(math.isnan(lo) or math.isnan(hi) for lo, hi in zip(ci_low, ci_high)):
        ax_main.fill_between(
            times,
            ci_low,
            ci_high,
            color='tab:blue',
            alpha=0.18,
            label='Confidence band',
        )

    if any(math.isfinite(val) for val in baseline_vals):
        ax_main.plot(
            times,
            baseline_vals,
            label=baseline_label,
            color='tab:orange',
            linestyle='--',
            linewidth=1.8,
        )

    if any(math.isfinite(val) for val in truth_vals):
        ax_main.plot(times, truth_vals, label=truth_label, color='black', linestyle='-.', linewidth=2.0, marker='x')

    if comparison_series:
        for label, series in comparison_series.items():
            comp_vals = _ensure_float_sequence(series, key=label)
            if len(comp_vals) != len(times):
                raise ValueError(f'Comparison series "{label}" has length {len(comp_vals)} but expected {len(times)}.')
            ax_main.plot(
                times,
                comp_vals,
                label=label,
                linestyle=':',
                linewidth=1.6,
            )

    base_title = 'DynamicRA trajectory and baseline comparison'
    info_bits = []
    if iterations is not None:
        info_bits.append(f'iterations={iterations}')
    if path_label is not None:
        info_bits.append(f'path={tuple(path_label)}')
    overall = metrics.get('overall', {})
    rmse_overall = overall.get('rmse_mean')
    abs_bias_overall = overall.get('abs_bias_mean')
    if rmse_overall is not None and math.isfinite(rmse_overall):
        info_bits.append(f'RMSE={rmse_overall:.3f}')
    if abs_bias_overall is not None and math.isfinite(abs_bias_overall):
        info_bits.append(f'|bias|={abs_bias_overall:.3f}')

    ax_main.set_ylabel('Estimate / benchmark')
    if info_bits:
        ax_main.set_title(f'{base_title}\n{" | ".join(info_bits)}')
    else:
        ax_main.set_title(base_title)
    ax_main.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    # Panel 2: Contribution diagnostics
    ax_contrib.axhline(0.0, color='black', linewidth=0.8, linestyle='--')

    if any(math.isfinite(val) for val in residual_vals):
        ax_contrib.plot(
            times,
            residual_vals,
            color='tab:green',
            marker='s',
            label='Residual correction',
        )
    if any(math.isfinite(val) for val in aug_vals):
        ax_contrib.plot(
            times,
            aug_vals,
            color='tab:purple',
            marker='^',
            label='Transition augmentation',
        )

    delta_vals = tuple(
        mu - base if math.isfinite(mu) and math.isfinite(base) else math.nan
        for mu, base in zip(mu_vals, baseline_vals)
    )
    if any(math.isfinite(val) for val in delta_vals):
        ax_contrib.plot(
            times,
            delta_vals,
            color='tab:red',
            linestyle=':',
            marker='o',
            label='Dynamic minus baseline',
        )

    if any(math.isfinite(val) for val in bias_vals):
        ax_contrib.plot(
            times,
            bias_vals,
            color='tab:olive',
            linestyle='--',
            marker='o',
            label='Dynamic minus truth',
        )
    if any(math.isfinite(val) for val in rmse_vals):
        ax_contrib.plot(
            times,
            rmse_vals,
            color='tab:brown',
            linestyle='-',
            marker='.',
            label='RMSE',
        )

    ax_contrib.set_ylabel('Contribution / error')
    ax_contrib.set_title('DynamicRA decomposition')
    handles, labels = ax_contrib.get_legend_handles_labels()
    if handles:
        ax_contrib.legend(
            handles,
            labels,
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    ax_contrib.set_xlabel('Time horizon')

    _annotate_event_lines(ax_main, events)
    _annotate_event_lines(ax_contrib, events, alpha=0.25, draw_labels=False)

    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_estimator_path_paper(
    metrics: Dict[str, Any],
    figsize: Tuple[float, float] = (6.0, 3.8),
    style: str = 'ticks',
    estimator_label: str = 'Estimate',
    ci_label: Optional[str] = None,
    ci_source: str = 'auto',
    color: str = 'tab:blue',
    marker: Optional[str] = None,
    linewidth: float = 2.2,
    ci_alpha: float = 0.18,
    xlabel: str = 'Time horizon',
    ylabel: str = 'Estimate',
    ylim: Optional[Tuple[float, float]] = None,
    grid: bool = True,
    title: Optional[str] = None,
    legend: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 100,
) -> plt.Figure:
    per_time = metrics['per_time']
    times = tuple(int(entry['t']) for entry in per_time)
    mean_hat = _ensure_float_sequence((entry['mean_hat'] for entry in per_time), key='mean_hat')

    ci_low_ls = []
    ci_high_ls = []
    for entry in per_time:
        boot_lo = entry['bootstrap_mean_ci_low']
        boot_hi = entry['bootstrap_mean_ci_high']
        ci_low_ls.append(boot_lo)
        ci_high_ls.append(boot_hi)
    ci_low = tuple(ci_low_ls)
    ci_high = tuple(ci_high_ls)

    if ci_label is None:
        ci_level = metrics['ci_level']
        if ci_level is not None:
            try:
                ci_label = f'{int(round(float(ci_level) * 100.0))}% CI'
            except (TypeError, ValueError):
                ci_label = 'CI'
        else:
            ci_label = 'CI'

    with sns.axes_style(style, rc=styles_kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        sns.despine(fig=fig)

        ax.spines['left'].set_linewidth(line_width)
        ax.spines['bottom'].set_linewidth(line_width)
        ax.tick_params(width=line_width)

        fig.set_dpi(dpi)

    ax.plot(
        times,
        mean_hat,
        label=estimator_label,
        color=color,
        linewidth=linewidth,
        marker=marker,
    )
    ax.tick_params(labelsize=27, labelleft=True)

    ax.fill_between(
        times,
        ci_low,
        ci_high,
        color=color,
        alpha=ci_alpha,
        linewidth=0.0,
        label=ci_label,
    )

    ax.set_xlabel(xlabel, fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.set_xticks(times)
    if ylim is not None:
        if len(ylim) != 2:
            raise ValueError('ylim must be a tuple of (lower, upper).')
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if grid:
        ax.grid(True, axis='both', linestyle='--', linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, fontsize=32)
    if legend:
        ax.legend(loc='best', fontsize=27)

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    return fig


def _annotate_event_lines(
    ax: plt.Axes,
    events: Optional[Iterable[Tuple[int, str]]],
    alpha: float = 0.4,
    draw_labels: bool = True,
) -> None:
    if not events:
        return
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    offset = 0.05 * span
    for time, label in events:
        ax.axvline(time, color='grey', linestyle=':', linewidth=1.0, alpha=alpha)
        if draw_labels and label:
            ax.text(
                time,
                y_max - offset,
                label,
                rotation=90,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize='small',
                color='grey',
                alpha=alpha + 0.2,
            )

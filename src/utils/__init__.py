import os
import random
import shutil
import textwrap
from dataclasses import fields, is_dataclass
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tabulate import tabulate
from tqdm import tqdm

BOLD = '\033[1m'
BLACK = '\033[30m'
MAG_BG = '\033[45m'
END = '\033[0m'


def cprint(element: str, **kwargs: dict) -> None:
    element = f'{BOLD}{MAG_BG}{BLACK} === {element} === {END}'
    if kwargs.get('end', False):
        print(element, end='')
    elif kwargs.get('sec', True):
        print('\n' + element)
    else:
        print(element)


def line() -> None:
    print('-' * 30)


def set_seed(seed: int = 42, use_gpu: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if use_gpu:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def status(status_bar: tqdm, msg: str) -> None:
    if status_bar is not None:
        status_bar.set_description_str(msg)
        status_bar.refresh()


def build_arrays(df: pd.DataFrame, cfg: Optional[DictConfig] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if cfg is None:
        time_col = 'date'
        unit_col = 'id'
        features_col = 'feature'
        val_col = 'value'
        Y_name = 'Y'
        W_name = 'W'
        X_names = None
    else:
        time_col = cfg.model.time
        unit_col = cfg.model.unit
        features_col = cfg.model.feature
        val_col = cfg.model.value
        Y_name = cfg.model.outcome
        W_name = cfg.model.treatment
        if cfg.model.get('covariates') is not None:
            X_names = list(cfg.model.covariates)
        else:
            X_names = None

    # `time` is expected to be an index (0..T-1) in the long-format dataframe.
    T = df[time_col].max() + 1
    ids = np.sort(df[unit_col].unique())
    n = len(ids)

    if X_names is None:
        all_features = list(df[features_col].unique())
        all_features.remove(Y_name)
        all_features.remove(W_name)
        X_names = all_features

    # Build arrays
    d = len(X_names)
    X = np.zeros((n, T, d), float)
    Y = np.zeros((n, T), float)
    W = np.zeros((n, T), int)
    id_index = {i: idx for idx, i in enumerate(ids)}
    for _, row in df.iterrows():
        tau = row[time_col]
        i = id_index[row[unit_col]]
        var = row[features_col]
        val = row[val_col]
        if var == Y_name:
            Y[i, tau] = val
        elif var == W_name:
            W[i, tau] = val
        elif var in X_names:
            k = X_names.index(var)
            X[i, tau, k] = val

    return X, Y, W, T


def count_all_W_patterns(data: pd.DataFrame, T: int) -> pd.DataFrame:
    # Create a list of dates from 0 to T-1
    dates = list(range(T))
    # Extract only W and use only the specified dates
    df_w = data.loc[(data['feature'] == 'W') & (data['date'].isin(dates)), ['id', 'date', 'value']]

    # Pivot id as rows and date as columns, fixing the order of dates
    wide = df_w.pivot(index='id', columns='date', values='value').reindex(columns=list(dates))

    # Target only ids with all 3 days present
    wide = wide.dropna(how='any')

    # Aggregate using the combination of columns (= W values for each date) as keys
    counts = wide.groupby(list(wide.columns)).size().reset_index(name='count').sort_values('count', ascending=False)

    # Rename columns for readability and add a pattern column
    rename_map = {d: f'W(d={d})' for d in dates}
    counts = counts.rename(columns=rename_map)
    pattern_cols = [f'W(d={d})' for d in dates]
    counts.insert(0, 'pattern', list(map(tuple, counts[pattern_cols].to_numpy())))

    return counts[['pattern', 'count'] + pattern_cols]


def _fmt_value(v: Any, float_sig: int = 6) -> str:
    if isinstance(v, float):
        return f'{v:.{float_sig}g}'
    if isinstance(v, (tuple, list)):
        open_b, close_b = ('(', ')') if isinstance(v, tuple) else ('[', ']')
        return open_b + ', '.join(_fmt_value(x, float_sig) for x in v) + close_b
    if isinstance(v, dict):
        items = ', '.join(f'{_fmt_value(k, float_sig)}: {_fmt_value(val, float_sig)}' for k, val in v.items())
        return '{' + items + '}'
    if isinstance(v, str):
        return repr(v)
    return str(v)


def format_config_box(
    obj: dict,
    title: str = 'Config',
    show_types: bool = False,
    width: Optional[int] = None,
    unicode_box: bool = True,
    *,
    max_val_width: int = 56,
    max_key_width: int = 24,
) -> str:
    if unicode_box:
        H, V = '─', '│'
        TL, TR, BL, BR = '┌', '┐', '└', '┘'
        LSEP, RSEP = '├', '┤'
        TSEP, BSEP = '┬', '┴'
    else:
        H, V = '-', '|'
        TL = TR = BL = BR = '+'
        LSEP = RSEP = '+'
        TSEP = BSEP = '+'

    items: list[tuple[str, Any]] = []
    if is_dataclass(obj):
        for f in fields(obj):
            items.append((f.name, getattr(obj, f.name)))
    elif isinstance(obj, dict):
        for k in obj:
            items.append((str(k), obj[k]))
    else:
        raise TypeError('format_config_box: Please provide `dataclass` or `dict`.')

    key_max = max((len(k) for k, _ in items), default=8)
    key_w = min(max(10, key_max), max_key_width)

    FIXED = 7

    if width is None:
        try:
            term_w = shutil.get_terminal_size((100, 20)).columns
        except Exception:
            term_w = 100
        width = min(term_w, FIXED + key_w + max_val_width)
    width = max(48, min(200, width))

    val_w = width - FIXED - key_w
    if val_w < 10:
        need = 10 - val_w
        shrinkable = max(0, key_w - 10)
        take = min(need, shrinkable)
        key_w -= take
        val_w = width - FIXED - key_w
        val_w = max(10, val_w)

    top_border = TL + H * (key_w + 2) + H * (val_w + 3) + TR
    header_sep = LSEP + H * (key_w + 2) + TSEP + H * (val_w + 2) + RSEP
    bottom_border = BL + H * (key_w + 2) + BSEP + H * (val_w + 2) + BR

    title_str = f' {title} '
    inside_total_w = width - 3
    left_inside_w = key_w + 2
    inside_total = title_str.center(inside_total_w, ' ')
    title_line = V + inside_total[:left_inside_w] + ' ' + inside_total[left_inside_w:] + V

    lines = [top_border, title_line, header_sep]
    for k, v in items:
        v_str = _fmt_value(v)
        if show_types:
            v_str = f'{v_str}    ({type(v).__name__})'
        wrapped = textwrap.wrap(v_str, width=val_w) or ['']
        lines.append(f'{V} {k:<{key_w}} {V} {wrapped[0]:<{val_w}} {V}')
        for cont in wrapped[1:]:
            lines.append(f'{V} {"":<{key_w}} {V} {cont:<{val_w}} {V}')

    lines.append(bottom_border)
    return '\n'.join(lines)


def print_cfg(obj: dict, **kwargs: Any) -> None:
    print(format_config_box(obj, **kwargs))


def tabulate_wide(
    wide: pd.DataFrame,
    value_name: str = 'values',
    index_name: Optional[str] = None,
    style: Literal['multirow', 'flatten', 'long'] = 'multirow',
    sep: str = ' / ',
    dropna_values: bool = False,
    tablefmt: str = 'psql',
    floatfmt: str = 'g',
) -> str:
    idx_name = index_name if index_name is not None else (wide.index.name or 'time')

    # --- long style ---
    if style == 'long':
        if not isinstance(wide.columns, pd.MultiIndex):
            # Works with single-level columns: Treat column names as a single level
            cols = [str(c) for c in wide.columns]
            df = wide.copy()
            df.columns = pd.MultiIndex.from_arrays([cols], names=['col'])
        else:
            df = wide

        levels = list(range(df.columns.nlevels))
        try:
            s = df.stack(level=levels, future_stack=True)
            long = s.rename(value_name).reset_index()
            if dropna_values:
                long = long.dropna(subset=[value_name])
        except TypeError:
            s = df.stack(level=levels, dropna=dropna_values)
            long = s.rename(value_name).reset_index()

        long = long.rename(columns={long.columns[0]: idx_name})
        # Rearrange columns into [index] + column levels + [value]
        col_level_names = [nm if nm is not None else f'level_{i}' for i, nm in enumerate(df.columns.names)]
        out = long[[idx_name] + col_level_names + [value_name]]
        return tabulate(out, headers='keys', tablefmt=tablefmt, showindex=False, floatfmt=floatfmt)

    # --- flatten style ---
    if style == 'flatten':
        out = wide.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = pd.Index(
                [sep.join(map(str, tup)) for tup in out.columns.to_flat_index()],
                dtype='object',
            )
        else:
            out.columns = out.columns.map(str)
        return tabulate(out, headers='keys', tablefmt=tablefmt, showindex=True, floatfmt=floatfmt)

    # --- multirow style ---
    if not isinstance(wide.columns, pd.MultiIndex):
        # Works with single-level columns: Treat column names as a single level
        return tabulate(wide, headers='keys', tablefmt=tablefmt, showindex=True, floatfmt=floatfmt)

    nlv = wide.columns.nlevels
    header_rows: List[List[str]] = []
    # Create header rows for each level
    for level in range(nlv):
        row = []
        row.append(idx_name if level == 0 else '')  # index column
        for col in wide.columns:
            row.append(str(col[level]))
        header_rows.append(row)

    # Data body (index values + each column value)
    body_rows: List[List[str]] = []
    for idx, rowvals in wide.iterrows():
        row = [str(idx)]
        # rowvals is 1D (corresponding to column MultiIndex)
        for v in rowvals:
            if pd.isna(v):
                row.append('nan')
            else:
                if isinstance(v, (float, np.floating)):
                    row.append(('{:' + floatfmt + '}').format(v))
                else:
                    row.append(str(v))
        body_rows.append(row)

    # Concatenate all and render as "header-less table"
    table = header_rows + body_rows
    # Clear column headers and use the created header_rows as actual data
    return tabulate(table, headers=[], tablefmt=tablefmt, showindex=False)


def configure_thread_limits() -> None:
    for name in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'BLIS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ.setdefault(name, '1')

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

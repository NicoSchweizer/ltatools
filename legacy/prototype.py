import pandas as pd
import numpy as np
import allantools as at
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def load_lta_file(file_path, cleanup=False):
    """
    Parse a HighFinesse .lta file into a DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the .lta file (tab-separated, cp1252 encoding, comma decimal).
    cleanup : bool, optional
        If True, drops rows where wavelength <= 0 (invalid readings). Default False.

    Returns
    -------
    pd.DataFrame
        Columns from the file plus:
        - ``Signal 1 Frequency  [THz]`` – computed as c / lambda
        - ``Time  [s]`` – Time [ms] converted to seconds
    """
    skip_rows = 0
    with open(file_path, 'r', encoding='cp1252', errors='ignore') as f:
        for i, line in enumerate(f):
            if "Time" in line:
                skip_rows = i
                break

    if skip_rows == 0:
        raise ValueError("Der Tabellen-Header mit 'Time' wurde in der Datei nicht gefunden.")

    df = pd.read_csv(
        file_path,
        skiprows=skip_rows,
        sep='\t',
        decimal=',',
        skipinitialspace=True,
        encoding='cp1252'
    )

    df.columns = df.columns.str.strip()

    if cleanup==True:
        df = df[df['Signal 1  Wavelength, vac.  [nm]'] > 0].copy()

    df['Signal 1 Frequency  [THz]'] = (sp.constants.c / (df['Signal 1  Wavelength, vac.  [nm]'] * 1e-9)) * 1e-12
    df['Time  [s]'] = df['Time  [ms]'] * 1e-3

    return df

def ad_plot(data, data_type='freq', rate=0.0, time_data=0, taus='all',
            title='Overlap Allan Deviation of frequency', ylabel=r'$\sigma(\tau)$ in Hz',
            xlabel=r'$\tau$ in s', fig_file="figures/oadev_1_freq.png", y_range='THz'):
    """
    Compute and plot the overlap Allan deviation (OADEV).

    Parameters
    ----------
    data : array-like
        Frequency or power data series.
    data_type : str, optional
        ``'freq'`` (default) – data is treated as frequency samples.
    rate : float, optional
        Sampling rate in Hz. If 0.0 (default), rate is derived from ``time_data``.
    time_data : array-like, optional
        Timestamps in **ms** – used to compute rate when ``rate == 0.0``.
    taus : str or array-like, optional
        Averaging times: ``'all'`` (default) or ``'octave'``.
    title : str, optional
        Plot title.
    ylabel : str, optional
        y-axis label (overridden by ``y_range``).
    xlabel : str, optional
        x-axis label.
    fig_file : str, optional
        File path to save the figure (PNG, 300 dpi).
    y_range : str, optional
        Unit for the y-axis. Scales OADEV accordingly.
        ``'THz'`` | ``'GHz'`` | ``'MHz'`` | ``'kHz'`` | ``'Hz'`` | ``'µW'``

    Returns
    -------
    tau : np.ndarray
        Averaging times in seconds.
    oadev : np.ndarray
        OADEV values in the unit selected by ``y_range``.
    """

    if rate == 0.0:
        time_data = time_data * 1e-3
        dt = time_data.diff().dropna()
        dt = dt[dt > 0]
        rate = 1 / np.mean(dt)

    (tau, oadev, _, _) = at.oadev(np.asarray(data), rate=rate, data_type=data_type, taus=taus)

    if y_range == 'THz':
        oadev = oadev
        ylabel = r'$\sigma(\tau)$ in THz'
    elif y_range == 'GHz':
        oadev = oadev * 1e3
        ylabel = r'$\sigma(\tau)$ in GHz'
    elif y_range == 'MHz':
        oadev = oadev * 1e6
        ylabel = r'$\sigma(\tau)$ in MHz'
    elif y_range == 'kHz':
        oadev = oadev * 1e9
        ylabel = r'$\sigma(\tau)$ in kHz'
    elif y_range == 'Hz':
        oadev = oadev * 1e12
        ylabel = r'$\sigma(\tau)$ in Hz'

    plt.figure(figsize=(8, 5))
    plt.loglog(tau, oadev, marker='x')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(fig_file, dpi=300)
    plt.show()
    return (tau, oadev)

def t_s_plot(data, fig_file, plot_type='freq', lines=False):
    """
    Dual-axis time-series plot: frequency or wavelength (left) and power (right).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame from ``load_lta_file`` or ``find_stable_segments``.
        Must contain ``Time  [ms]``, ``Signal 1 Power  [µW]``, and either
        ``Signal 1 Frequency  [THz]`` or ``Signal 1  Wavelength, vac.  [nm]``.
    fig_file : str
        File path to save the figure (PNG, 300 dpi).
    plot_type : str, optional
        ``'freq'`` (default) – plot frequency on the left axis.
        ``'wl'`` – plot wavelength on the left axis.
    lines : bool, optional
        If True, connect data points with lines (marker ``'x-'``). Default False.

    Returns
    -------
    None
    """
    fmt = 'x'
    if lines==True:
        fmt = 'x-'

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)

    space_y_pow = 3

    ax2.errorbar(
        data['Time  [ms]'] * 1e-3,
        data['Signal 1 Power  [µW]'] * 1e-3,
        fmt=fmt,
        label='Power (mW)',
        color='tab:orange',
        zorder=1
    )
    ax2.set_ylabel('Power (mW)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(
        np.percentile(data['Signal 1 Power  [µW]'] * 1e-3, 0) - space_y_pow,
        np.percentile(data['Signal 1 Power  [µW]'] * 1e-3, 100) + space_y_pow
    )

    if plot_type == 'wl':
        space_y = 0.002
        ax1.errorbar(
            data['Time  [ms]'] * 1e-3,
            data['Signal 1  Wavelength, vac.  [nm]'],
            fmt=fmt,
            label='Wavelength (nm)',
            color='tab:blue',
            zorder=3
        )
        ax1.set_title('Wavelength and Power over time')
        ax1.set_ylabel('Wavelength (nm)', color='tab:blue')
        ax1.set_ylim(
            np.percentile(data['Signal 1  Wavelength, vac.  [nm]'], 0) - space_y,
            np.percentile(data['Signal 1  Wavelength, vac.  [nm]'], 100) + space_y
        )

    elif plot_type == 'freq':
        space_y = 1e-1
        ax1.errorbar(
            data['Time  [ms]'] * 1e-3,
            data['Signal 1 Frequency  [THz]'],
            fmt=fmt,
            label='Frequency (THz)',
            color='tab:blue',
            zorder=3
        )
        ax1.set_title('Frequency and Power over time')
        ax1.set_ylabel('Frequency (THz)', color='tab:blue')
        ax1.set_ylim(
            np.percentile(data['Signal 1 Frequency  [THz]'], 0) - space_y,
            np.percentile(data['Signal 1 Frequency  [THz]'], 100) + space_y
        )

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xlabel('Time (s)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines2 + lines1, labels2 + labels1, loc='best')
    legend.set_zorder(10)

    ax1.grid(True, which="both", ls="--")
    fig.tight_layout()
    plt.savefig(fig_file, dpi=300)

    plt.ion()

    plt.show()
    return

def lta_to_adev(file_path, fig_file, taus='all', plot_type='wl', y_range='THz', cleanup=False):
    """
    Load an .lta file and directly plot its overlap Allan deviation.

    Convenience wrapper around ``load_lta_file`` + ``ad_plot``.

    Parameters
    ----------
    file_path : str
        Path to the .lta file.
    fig_file : str
        File path to save the figure (PNG, 300 dpi).
    taus : str or array-like, optional
        Averaging times passed to ``ad_plot``: ``'all'`` (default) or ``'octave'``.
    plot_type : str, optional
        Which channel to analyse: ``'freq'`` | ``'wl'`` | ``'pow'``. Default ``'wl'``.
    y_range : str, optional
        Unit for the y-axis (see ``ad_plot``). Default ``'THz'``.
    cleanup : bool, optional
        Passed to ``load_lta_file`` – removes invalid (≤0) wavelength rows.

    Returns
    -------
    None
    """
    df = load_lta_file(file_path, cleanup=cleanup)

    ylabel = ''

    if plot_type == 'wl':
        ad_plot(data=df['Signal 1  Wavelength, vac.  [nm]'], data_type='freq', time_data=df['Time  [ms]'], taus=taus, fig_file=fig_file,
                title='Overlap Allan Deviation of wavelength', ylabel=r'$\sigma(\tau)$ in nm', xlabel=r'$\tau$ in s')
    elif plot_type == 'pow':
        if y_range == 'uW':
            yrange_factor = 1
            ylabel = r'$\sigma(\tau)$ in µW'
        elif y_range == 'mW':
            yrange_factor = 1e-3
            ylabel = r'$\sigma(\tau)$ in mW'
        elif y_range == 'W':
            yrange_factor = 1e-6
            ylabel = r'$\sigma(\tau)$ in W'
        ad_plot(data=df['Signal 1 Power  [µW]'] * yrange_factor, data_type='freq', time_data=df['Time  [ms]'], taus=taus, fig_file=fig_file,
                title='Overlap Allan Deviation of power', ylabel=ylabel, xlabel=r'$\tau$ in s')
    elif plot_type == 'freq':
        if y_range == 'THz':
            ylabel = r'$\sigma(\tau)$ in THz'
        elif y_range == 'GHz':
            ylabel = r'$\sigma(\tau)$ in GHz'
        elif y_range == 'MHz':
            ylabel = r'$\sigma(\tau)$ in MHz'
        elif y_range == 'kHz':
            ylabel = r'$\sigma(\tau)$ in kHz'
        elif y_range == 'Hz':
            ylabel = r'$\sigma(\tau)$ in Hz'
        ad_plot(data=df['Signal 1 Frequency  [THz]'], data_type='freq', time_data=df['Time  [ms]'], taus=taus, fig_file=fig_file,
                title='Overlap Allan Deviation of frequency', ylabel=ylabel, xlabel=r'$\tau$ in s', y_range=y_range)
    return

def lta_to_t_s(file_path, fig_file, plot_type='freq', cleanup=False, lines=False):
    """
    Load an .lta file and directly plot the time series.

    Convenience wrapper around ``load_lta_file`` + ``t_s_plot``.

    Parameters
    ----------
    file_path : str
        Path to the .lta file.
    fig_file : str
        File path to save the figure (PNG, 300 dpi).
    plot_type : str, optional
        ``'freq'`` (default) – plot frequency vs. time.
        ``'wl'`` – plot wavelength vs. time.
    cleanup : bool, optional
        Passed to ``load_lta_file`` – removes invalid (≤0) wavelength rows.
    lines : bool, optional
        If True, connect data points with lines. Default False.

    Returns
    -------
    None
    """
    df = load_lta_file(file_path, cleanup=cleanup)
    t_s_plot(data=df, fig_file=fig_file, plot_type=plot_type, lines=lines)
    return

def find_stable_segments(df, freq_col='Signal 1 Frequency  [THz]', n=2, threshold=None, jump_factor=20,
                         trim_left=2, trim_right=0):
    """
    Return the n longest frequency-stable segments of df, free of mode hops.

    Parameters
    ----------
    df          : DataFrame returned by load_lta_file
    freq_col    : column to detect jumps on
    n           : number of longest segments to return
    threshold   : jump size in freq_col units; auto-detected if None
    jump_factor : multiplier on median step size for auto threshold
    trim_left   : number of points to drop from the start of each segment (default 2)
    trim_right  : number of points to drop from the end of each segment (default 0)

    Returns
    -------
    list of DataFrames, longest first; row index and Time  [ms] both reset to 0
    """
    diffs = df[freq_col].diff().abs()

    if threshold is None:
        typical = diffs[diffs > 0].median()
        threshold = jump_factor * typical

    segment_id = (diffs > threshold).cumsum()

    segments = []
    for _, grp in df.groupby(segment_id):
        end = len(grp) - trim_right if trim_right > 0 else len(grp)
        grp = grp.iloc[trim_left:end]
        if len(grp) == 0:
            continue
        grp = grp.reset_index(drop=True)
        grp = grp.assign(**{'Time  [ms]': grp['Time  [ms]'] - grp['Time  [ms]'].iloc[0]})
        segments.append(grp)

    segments.sort(key=len, reverse=True)
    return segments[:n]

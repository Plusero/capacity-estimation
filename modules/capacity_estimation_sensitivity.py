import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.capacity_estimation import capacity_estimation_base_load
from modules.f_common_script import calc_percentage_error, calc_mean_percentage_error, calc_r2_score


def sensitivity_analysis_irradiance_threshold(df_tmp: pd.DataFrame, real_capacity_tmp: np.ndarray, fig_name: str, base_load_correction_factor: float = None, metric: str = 'PE'):
    """
    Performs sensitivity analysis on irradiance thresholds and creates a heatmap visualization.

    This function analyzes how different combinations of day and night irradiance thresholds
    affect the capacity estimation accuracy. The values of the heatmap are given by the specified metric.

    Parameters:
    -----------
    df_tmp : pandas.DataFrame
        DataFrame containing the power and irradiance measurements.
    real_capacity_tmp : numpy.ndarray or array-like
        Array of actual PV system capacities.
    fig_name : str
        File path where the resulting heatmap will be saved.
    base_load_correction_factor : float, optional
        Correction factor for base load estimation. Default is None.
    metric : str, optional
        Performance metric to use. Options are:
        - 'PE': Percentage Error
        - 'MPE': Mean Percentage Error
        - 'R2': R-squared score
        Default is 'PE'.

    Returns:
    None
    """
    list_irradiance_thresholds_noon = [
        10, 20, 30, 40] + list(range(50, 550, 50))
    list_irradiance_thresholds_night = [0.01, 0.1, 1, 10]
    net_cols_without_total_net = [
        col for col in df_tmp.columns if col.endswith('_net') and col != 'total_net']
    # store the error rates in a matrix
    error_rates = np.zeros(
        (len(list_irradiance_thresholds_noon), len(list_irradiance_thresholds_night)))
    for i, irradiance_threshold_noon in enumerate(list_irradiance_thresholds_noon):
        for j, irradiance_threshold_night in enumerate(list_irradiance_thresholds_night):
            cap_est_based_load = capacity_estimation_base_load(
                df_tmp, net_cols_without_total_net, irradiance_threshold_noon, irradiance_threshold_night, base_load_correction_factor=base_load_correction_factor)
            capacity_based_load = cap_est_based_load.estimate_capacity()
            if metric == 'PE':
                error_rates[i, j] = calc_percentage_error(
                    real_capacity_tmp, capacity_based_load)
            elif metric == 'MPE':
                error_rates[i, j] = calc_mean_percentage_error(
                    real_capacity_tmp, capacity_based_load)
            elif metric == 'R2':
                error_rates[i, j] = calc_r2_score(
                    real_capacity_tmp, capacity_based_load)
    if metric == 'R2':
        ticks = np.linspace(-1, 1, 11)
        v_min_n_max = [-1, 1]
        cmap = 'rocket'
        label = '$\mathrm{R}^2$'
    if metric == 'PE':
        ticks = np.linspace(-100, 100, 11)
        v_min_n_max = [-100, 100]
        cmap = 'RdBu_r'
        label = 'Percentage Error ($\%$)'
    if metric == 'MPE':
        ticks = np.linspace(-100, 100, 11)
        v_min_n_max = [-100, 100]
        cmap = 'RdBu_r'
        label = 'Mean Percentage Error ($\%$)'
    # Plot heatmap
    plt.style.use(['science'])
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        error_rates,
        cmap=cmap,
        center=0,
        linewidths=0.1,
        vmin=v_min_n_max[0],
        vmax=v_min_n_max[1],
        xticklabels=list_irradiance_thresholds_night,
        yticklabels=list_irradiance_thresholds_noon,
        annot=True,
        cbar_kws={
            'label': label,
            'ticks': ticks
        }
    )
    # Configure axis labels
    plt.xlabel('$I_{night}$ (W/m$^2$)')
    plt.ylabel('$I_{day}$ (W/m$^2$)')
    # Rotate y-axis labels to horizontal
    plt.yticks(rotation=0)
    # Save and display
    plt.tight_layout()
    plt.savefig(fig_name, format='pdf')
    plt.show()
    return None

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import re
import scienceplots
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

# Section:Global variables
my_seed = 42
np.random.seed(my_seed)


# Section:Functions for plotting
def add_trendline(x: np.ndarray, y: np.ndarray, ax: plt.Axes) -> None:
    """
    Adds a quadratic trendline to a given plot.

    This function sorts the input data, fits a second-order polynomial
    to the sorted data, and plots the resulting trendline on the provided
    Axes object.

    Parameters:
    x (np.ndarray or array-like): The x-coordinates of the data points.
    y (np.ndarray or array-like): The y-coordinates of the data points.
    ax (matplotlib.axes.Axes): The Axes object on which to plot the trendline.

    Returns:
    None
    """
    # Sort x and y values to ensure a continuous line
    # convert x and y to numpy array
    x = np.array(x)
    y = np.array(y)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Fit polynomial and create trendline
    z = np.polyfit(x_sorted, y_sorted, 2)
    p = np.poly1d(z)
    ax.plot(x_sorted, p(x_sorted), "--", color='red', linewidth=2)


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, fig_name: str = '../figs/actual_vs_predicted.pdf', xlabel: str = 'Actual', ylabel: str = 'Estimated', trendline: bool = True) -> None:
    """
    Plots the actual vs. predicted values and optionally a trendline.
    This function creates a scatter plot of the actual vs. predicted values,
    draws a reference line of y=x, and optionally adds a quadratic trendline.
    The plot is saved as a PDF file.

    Parameters:
    y_test (array-like): The actual values.
    y_pred (array-like): The predicted values.
    fig_name (str): The file path where the plot will be saved. Default is '../figs/actual_vs_predicted.pdf'.
    xlabel (str): The label for the x-axis. Default is 'Actual'.
    ylabel (str): The label for the y-axis. Default is 'Estimated'.
    trendline (bool): Whether to add a trendline to the plot. Default is True.

    Returns:
    None
    """
    # plot the actual vs predicted, with a line of y=x
    plt.style.use(['science'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [
        y_test.min(), y_test.max()], 'k-', lw=2)
    if trendline:
        add_trendline(y_test, y_pred, ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(fig_name, format='pdf')
    plt.show()


def df_common_xylabel_plot(df: pd.DataFrame, y: list[str], doy_start: int = 0, doy_end: int = 366, year: list[int] = [2017, 2018, 2019], xlabel: str = 'Datetime', ylabel: str = "example ylabel", layout: tuple = (7, 4), subplots: bool = True, figsize: tuple = (20, 20)) -> None:
    """
    Plots specified columns of a DataFrame over a given range of time.

    This function filters the DataFrame based on the day of the year and the specified years,
    and then plots the specified columns against the 'datetime' column. It supports creating
    subplots and adds common x and y labels to the figure.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to plot.
    y (str or list of str): The column(s) to plot on the y-axis.
    doy_start (int): The starting day of the year for filtering. Default is 0.
    doy_end (int): The ending day of the year for filtering. Default is 366.
    year (list of int): The years to include in the plot. Default is [2017, 2018, 2019].
    xlabel (str): The label for the x-axis. Default is 'Datetime'.
    ylabel (str): The label for the y-axis. Default is 'example ylabel'.
    layout (tuple): The layout of the subplots (rows, columns). Default is (7, 4).
    subplots (bool): Whether to create subplots for each column. Default is True.
    figsize (tuple): The size of the figure. Default is (20, 20).

    Returns:
    None
    """
    # make a list of columns to plot, excluding the columns of datetime and Timestamp
    axes = df[(df['doy'] >= doy_start) & (df['doy'] <= doy_end) & (df['year'].isin(year))].plot(
        x='datetime', y=y, subplots=subplots, figsize=figsize, layout=layout, sharex=True, xlabel='')
    if subplots == True:
        fig = axes[0, 0].get_figure()
    else:
        fig = axes.get_figure()  # to avoid the problem of "'Axes' object is not subscriptable"
    # add a common y label
    fig.text(0.08, 0.5, ylabel, va='center', rotation='vertical', size=20)
    # add a common x label
    fig.text(0.5, 0.0, xlabel, ha='center', size=20)

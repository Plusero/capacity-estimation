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


# Section:Global variables
my_seed = 42
np.random.seed(my_seed)


# Section:Functions for plotting
def add_trendline(x, y, ax):
    # Sort x and y values to ensure a continuous line
    # convert x and y to numpy array
    x = np.array(x)
    y = np.array(y)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Fit polynomial and create trendline
    z = np.polyfit(x_sorted, y_sorted, 5)
    p = np.poly1d(z)
    ax.plot(x_sorted, p(x_sorted), "--", color='red', linewidth=2)


def plot_actual_vs_predicted(y_test, y_pred, fig_name='../figs/actual_vs_predicted.pdf') -> None:
    # plot the actual vs predicted, with a line of y=x
    plt.style.use(['science'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [
        y_test.min(), y_test.max()], 'k-', lw=2)
    add_trendline(y_test, y_pred, ax)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    plt.savefig(fig_name, format='pdf')
    plt.show()

# check the 'normalized_total_pv_gen_correct' during date 2022-07-01 to 2022-07-31


def df_common_xylabel_plot(df, y, doy_start=0, doy_end=366, year=[2017, 2018, 2019], xlabel='Datetime', ylabel="example ylabel", layout=(7, 4), subplots=True, figsize=(20, 20)):
    # make a list to plot, excluding the columns of datetime and Timestamp
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

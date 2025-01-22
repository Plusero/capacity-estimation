import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates


class data_explorer():
    def __init__(self, pv_data: pd.DataFrame, con_data: pd.DataFrame):
        self.pv_data = pv_data
        self.con_data = con_data
        self.net_data = pd.DataFrame()
        self.ams_data = pd.DataFrame()
        self.missing_rate_pv = None
        self.missing_rate_con = None
        self.missing_rate_avg_pv = None
        self.missing_rate_avg_con = None

    def explore_data(self, data: pd.DataFrame):
        raise NotImplementedError("explore_data not implemented")

    def move_cols_to_front(self, cols_to_move: list, cols_to_keep: list, data_to_move: str = 'all') -> None:
        if data_to_move.lower() not in ['pv', 'con', 'net', 'all']:
            raise ValueError(
                "data_to_move must be 'pv', 'con', 'net', or 'all'")
        if data_to_move.lower() in ['pv', 'all']:
            self.pv_data = self.move_cols(
                cols_to_move, cols_to_keep, self.pv_data)
        if data_to_move.lower() in ['con', 'all']:
            self.con_data = self.move_cols(
                cols_to_move, cols_to_keep, self.con_data)
        if data_to_move.lower() in ['net', 'all']:
            self.net_data = self.move_cols(
                cols_to_move, cols_to_keep, self.net_data)

    def move_cols(self, cols_to_move: list, cols_to_keep: list, data_to_move: pd.DataFrame) -> pd.DataFrame:
        return data_to_move[cols_to_keep + cols_to_move +
                            [col for col in data_to_move.columns if col not in cols_to_move + cols_to_keep]]

    def unix_time_to_datetime(self) -> None:
        # convert the time column from Unix time to datetime.
        # Note that the Unix timestamp was in seconds, so we use unit='s'.
        self.pv_data['datetime'] = pd.to_datetime(
            self.pv_data['Timestamp'], origin='unix', unit='s')
        self.con_data['datetime'] = pd.to_datetime(
            self.con_data['Timestamp'], origin='unix', unit='s')
        # move the datetime column to the front
        cols_to_move = ['datetime']
        cols_to_keep = ['Timestamp']
        self.move_cols_to_front(cols_to_move, cols_to_keep, 'pv')
        self.move_cols_to_front(cols_to_move, cols_to_keep, 'con')

    def calculate_missing_rate(self) -> pd.Series:
        self.missing_rate_pv = self.pv_data.isnull().sum() / \
            self.pv_data.shape[0]
        self.missing_rate_con = self.con_data.isnull().sum() / \
            self.con_data.shape[0]
        self.missing_rate_avg_pv = self.missing_rate_pv.mean()
        self.missing_rate_avg_con = self.missing_rate_con.mean()
        # combine the missing rate of pv and con
        self.missing_rate = pd.concat(
            [self.missing_rate_pv, self.missing_rate_con], axis=1)
        self.missing_rate.columns = [
            'PV missing rate', 'Consumption missing rate']
        # check if the missing rate of pv and con are the same
        self.missing_rate_same = (self.missing_rate['PV missing rate'] ==
                                  self.missing_rate['Consumption missing rate']).all()
        return self.missing_rate_pv, self.missing_rate_con

    def calc_monthly_data_availability(self):
        # calculate the monthly data availability
        # the shape should be (24,27) for 24 months and 27 households
        # Get household columns (excluding datetime-related columns)
        household_cols = [col for col in self.pv_data.columns
                          if col not in ['datetime', 'Timestamp', 'HoD', 'dow', 'doy', 'month', 'year']]
        # add a column as year-month
        self.pv_data['year-month'] = self.pv_data['year'].astype(
            str) + '-' + self.pv_data['month'].astype(str).str.zfill(2)
        # str.zfill(2) is used to pad single-digit months with a leading zero.
        # "2017-1" becomes "2017-01"
        # "2017-2" becomes "2017-02"
        # Count non-null values for each month-year combination
        monthly_data = self.pv_data.groupby(['year-month'])[
            household_cols].count()
        # normalize the monthly data by the maximum possible readings per month
        monthly_data_availability = monthly_data / monthly_data.max()
        # transpose the dataframe
        monthly_data_availability = monthly_data_availability.transpose()
        # convert to percentage
        monthly_data_availability = monthly_data_availability * 100
        self.monthly_data_availability = monthly_data_availability
        print(
            f'shape of monthly_data_availability: {monthly_data_availability.shape}')
        return monthly_data_availability

    def data_availability_heat_map(self):
        # Random availability percentages
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(self.monthly_data_availability,
                         cmap='YlGnBu',
                         cbar_kws={
                             'label': 'Data availability rate [%]'},
                         linewidths=0.1,
                         linecolor='gray'
                         )
        # You can use the following to annotate the heatmap
        # annot=True,
        # annot_kws={'size': 10}
        # Adjust the colorbar label size
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(16)  # Set label size

        # remove the 'year' component from the x-axis labels
        ax.set_xticklabels([tick.get_text().split('-')[1]
                            for tick in ax.get_xticklabels()])
        # set the rotation of xticklabels to 0 degrees
        plt.xticks(fontsize=10, rotation=0, fontname='DejaVu Sans Mono')
        plt.yticks(fontsize=10, fontname='DejaVu Sans Mono')
        # place text [2017,2018,2019] at the bottom of the plot,
        # respectively at 20%, 60%, and 95% of the x-axis
        num_months = len(ax.get_xticklabels())
        # get the length of the y-axis
        y_length = ax.get_ylim()[1] - ax.get_ylim()[0]
        year_position = y_length * -1 * 1.05
        ax.text(x=num_months * 0.2, y=year_position, s='2017',
                fontsize=12, fontname='DejaVu Sans Mono')
        ax.text(x=num_months * 0.6, y=year_position, s='2018',
                fontname='DejaVu Sans Mono', fontsize=12)
        ax.text(x=num_months * 0.95, y=year_position, s='2019',
                fontname='DejaVu Sans Mono', fontsize=12)
        plt.ylabel('Household', fontname='DejaVu Sans Mono', fontsize=16)
        plt.xlabel('Time', fontname='DejaVu Sans Mono',
                   fontsize=16, labelpad=20)
        plt.tight_layout()
        # save the figure under folder "figs"
        plt.savefig(fname='../figs/data_availability_heat_map.pdf',
                    format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        # remove the year-month column
        self.pv_data = self.pv_data.drop(columns=['year-month'])

    def calculate_net_load(self) -> None:
        # To get net load, distract PV generation from consumption
        # drop the datetime and timestamp column
        self.net_data = self.con_data.drop(columns=['datetime', 'Timestamp'])
        # subtract the PV generation from the consumption
        self.net_data = self.net_data.sub(self.pv_data.drop(
            columns=['datetime', 'Timestamp']))
        # add the datetime and timestamp column back
        self.net_data['datetime'] = self.con_data['datetime']
        self.net_data['Timestamp'] = self.con_data['Timestamp']
        # move the datetime column to the front
        cols_to_move = ['datetime', 'Timestamp']
        cols_to_keep = []
        self.move_cols_to_front(cols_to_move, cols_to_keep, 'net')

    def add_datetime_columns(self) -> None:
        # for all dataframes
        # add hour, day of week, day of year, month,and year as columns, for easier data manipulation and further analysis
        # Define datetime features to extract
        dt_features = {
            'HoD': lambda x: x.dt.hour,
            'dow': lambda x: x.dt.dayofweek,
            'doy': lambda x: x.dt.dayofyear,
            'month': lambda x: x.dt.month,
            'year': lambda x: x.dt.year
        }
        # Add features to PV and consumption data
        for col, func in dt_features.items():
            self.pv_data.loc[:, col] = func(self.pv_data['datetime'])
            self.con_data.loc[:, col] = func(self.con_data['datetime'])
            self.net_data.loc[:, col] = func(self.net_data['datetime'])
        # Reorder columns
        cols_to_move = list(dt_features.keys())
        cols_to_keep = ['datetime', 'Timestamp']
        self.move_cols_to_front(cols_to_move, cols_to_keep, 'pv')
        self.move_cols_to_front(cols_to_move, cols_to_keep, 'con')
        self.move_cols_to_front(cols_to_move, cols_to_keep, 'net')

    def merge_data(self) -> None:
        # merge the three dataframes, rename the columns with suffixes
        # First, rename the columns of the three dataframes, except the time related columns
        cols_time_related = ['datetime', 'Timestamp',
                             'HoD', 'dow', 'doy', 'month', 'year']
        cols_time_not_related = [
            col for col in self.pv_data.columns if col not in cols_time_related]
        # Second, merge the three dataframes, keep the time related columns
        self.ams_data = pd.merge(
            self.pv_data, self.con_data, on=cols_time_related, how='inner', suffixes=('', '_con'))
        self.ams_data = pd.merge(self.ams_data, self.net_data,
                                 on=cols_time_related, how='inner', suffixes=('_pv', '_net'))

    def df_common_xylabel_plot(self, df, y, doy_start=0, doy_end=366, year=[2017, 2018, 2019], xlabel='Datetime', ylabel="example ylabel", layout=(7, 4), subplots=True, figsize=(20, 20)):
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

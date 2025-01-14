import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from crepes import WrapRegressor
from crepes.extras import margin, DifficultyEstimator, MondrianCategorizer
from lightgbm import LGBMRegressor
import seaborn as sns
import scienceplots
from matplotlib.dates import DateFormatter, DayLocator
from modules.data_split import split_data


class prob_regressor():
    def __init__(self,  original_data_part1: pd.DataFrame, original_data_part2: pd.DataFrame, random_seed: int = 42, list_of_features: list = None, y_col_name: str = 'total_pv_gen_normalized'):
        self.random_seed = random_seed
        self.data_part1 = original_data_part1
        self.data_part2 = original_data_part2
        self.list_of_features = list_of_features
        self.y_col_name = y_col_name
        self.irradiance_th_night = 0.01
        self.y_pred_mlr = None
        self.y_pred_rfr = None
        self.y_pred_lgb = None
        self.lgb = None
        self.learner_prop = None
        self.legends = ['Actual', "Predicted", "5\% CI", "95\% CI"]
        self.mc = None
        self.de = None
        self.cols_to_plot_cps_mond = None
        self.cols_to_plot_cps_mond_knn = None
        self.split_data()
        # fit the lgb model and yield point predictions
        self.fit_lgb()
        self.predict_lgb()
        # initialize the difficulty estimator and the mondrian categorizer
        self.difficulty_estimator(my_k=12)
        self.mond_categorizer(no_bins=6)
        # use the science plot style
        plt.style.use(['science'])

    def split_data(self):
        # It is essential to have exchangeability between calibration and test set
        # 2017-04-02 to 2018-09-22 as train
        self.train_data = self.data_part1[(self.data_part1['datetime'] >= '2017-04-02')
                                          & (self.data_part1['datetime'] <= '2018-09-22')]
        # 2018-09-23 to 2018-12-22 as calibration
        # autumn equinox to winter solstice
        self.cal_data = self.data_part1[(self.data_part1['datetime'] >= '2018-09-23') &
                                        (self.data_part1['datetime'] <= '2018-12-22')]
        # 2018-12-23 to 2019-03-10 06:00:00 as test
        # winter solstice to spring equinox
        self.test_data = self.data_part2[(self.data_part2['datetime'] >= '2018-12-23')
                                         & (self.data_part2['datetime'] <= '2019-03-10 06:00:00')]

        # Filter data where irradiance is greater than or equal to self.irradiance_th_night
        self.train_data = self.train_data[self.train_data['irradiance']
                                          >= self.irradiance_th_night]
        self.cal_data = self.cal_data[self.cal_data['irradiance']
                                      >= self.irradiance_th_night]
        self.test_data = self.test_data[self.test_data['irradiance']
                                        >= self.irradiance_th_night]

        self.X_prop_train = self.train_data[self.list_of_features]
        self.X_cal = self.cal_data[self.list_of_features]
        self.X_test = self.test_data[self.list_of_features]
        self.X_train = np.concatenate([self.X_prop_train, self.X_cal], axis=0)
        self.y_prop_train = self.train_data[self.y_col_name]
        self.y_cal = self.cal_data[self.y_col_name]
        self.y_test = self.test_data[self.y_col_name]
        self.y_train = np.concatenate([self.y_prop_train, self.y_cal], axis=0)
        # convert y_test to numpy array
        self.y_test = np.array(self.y_test)
        # check shapes
        print(f'X_train.shape: {self.X_train.shape}, X_test.shape: {self.X_test.shape}, X_prop_train.shape: {self.X_prop_train.shape}, X_cal.shape: {self.X_cal.shape}')

    def use_all_features(self):
        self.use_selected_features(self.list_of_features)

    def use_selected_features(self):
        self.use_certain_features(self.selected_features)
        print(f'Using selected features: {self.selected_features}')

    def use_certain_features(self, certain_features: list):
        self.X_prop_train = self.X_prop_train[certain_features]
        self.X_cal = self.X_cal[certain_features]
        self.X_test = self.X_test[certain_features]
        self.X_train = np.concatenate([self.X_prop_train, self.X_cal], axis=0)

    def mond_categorizer(self, no_bins) -> MondrianCategorizer:
        mc = MondrianCategorizer()
        mc.fit(X=self.X_prop_train, f=self.get_values,
               no_bins=no_bins)
        self.mc = mc

    def difficulty_estimator(self, my_k):
        de_knn = DifficultyEstimator()
        de_knn.fit(X=self.X_prop_train, k=my_k, scaler=True,
                   y=self.y_prop_train, beta=0.01)
        self.de = de_knn

    def fit_lgb(self):
        self.lgb = WrapRegressor(LGBMRegressor(
            n_jobs=-1, n_estimators=100, random_state=self.random_seed))
        self.lgb.fit(self.X_train, self.y_train)
        self.learner_prop = self.lgb.learner

    def predict_lgb(self):
        self.data_part2['total_pv_gen_normalized_pred_lgb'] = None
        self.data_part2['total_pv_gen_normalized_pred_lgb'] = self.lgb.predict(
            self.data_part2[self.list_of_features])
        # when irradiance is less than self.irradiance_th_night, set the predicted value to 0
        self.data_part2.loc[(self.data_part2['irradiance'] < self.irradiance_th_night),
                            'total_pv_gen_normalized_pred_lgb'] = 0

    def get_values(self, X):
        # The function get_values(X) is returning X[:, 0], which takes only the first column of X. Ensure that:
        # X is indeed a NumPy array or some data structure that supports indexing with [:,0].
        # If X is not a NumPy array, this operation could raise an error.
        # convert X to numpy array
        X = np.array(X)
        # returns only the first column of X
        # where the first column represents a significant feature derived from the Laplace matrix.
        # Leading eigenvalue from Laplace matrix
        return X[:, 0]

    def cps_mond_knn(self, confidence):
        lgb_cps_mond_knn = WrapRegressor(self.learner_prop)
        lgb_cps_mond_knn.calibrate(self.X_cal, self.y_cal,
                                   cps=True, mc=self.mc, de=self.de, seed=self.random_seed)
        # save the prediction interval with the following format:
        # cps_mond_knn_ci{confidence}_lower, cps_mond_knn_ci{confidence}_upper
        cps_mond_knn_cp_int = lgb_cps_mond_knn.predict_int(
            self.data_part2[self.list_of_features], y_min=0, y_max=1, confidence=confidence)
        cps_mond_knn_cp_lower = cps_mond_knn_cp_int[:, 0]
        cps_mond_knn_cp_upper = cps_mond_knn_cp_int[:, 1]
        # add the probabilities to the dataframe
        confidence_disp = int(confidence*100)
        # self.data_part2[f'cps_mond_knn_ci{confidence_disp}_lower'] = cps_mond_knn_cp_lower
        # self.data_part2[f'cps_mond_knn_ci{confidence_disp}_upper'] = cps_mond_knn_cp_upper
        # Create a dictionary of new columns
        new_columns = {
            f'cps_mond_knn_ci{confidence_disp}_lower': cps_mond_knn_cp_lower,
            f'cps_mond_knn_ci{confidence_disp}_upper': cps_mond_knn_cp_upper
        }

        # Add all columns at once using pd.concat
        self.data_part2 = pd.concat(
            [self.data_part2, pd.DataFrame(new_columns)], axis=1)

        # when irradiance is less than self.irradiance_th_night, set the predicted value to 0
        self.data_part2.loc[(self.data_part2['irradiance'] < self.irradiance_th_night),
                            f'cps_mond_knn_ci{confidence_disp}_lower'] = 0
        self.data_part2.loc[(self.data_part2['irradiance'] < self.irradiance_th_night),
                            f'cps_mond_knn_ci{confidence_disp}_upper'] = 0

    def cps_mond(self,  confidence):
        mc = self.mc
        lgb_cps_mond = WrapRegressor(self.learner_prop)
        lgb_cps_mond.calibrate(self.X_cal, self.y_cal,
                               cps=True, mc=mc, seed=self.random_seed)
        # save the prediction interval with the following format:
        # cps_mond_cp{confidence}_lower, cps_mond_cp{confidence}_upper
        cps_mond_cp_int = lgb_cps_mond.predict_int(
            self.data_part2[self.list_of_features], y_min=0, y_max=1, confidence=confidence)
        cps_mond_cp_lower = cps_mond_cp_int[:, 0]
        cps_mond_cp_upper = cps_mond_cp_int[:, 1]
        # add the probabilities to the dataframe
        confidence_disp = int(confidence*100)
        self.data_part2[f'cps_mond_ci{confidence_disp}_lower'] = cps_mond_cp_lower
        self.data_part2[f'cps_mond_ci{confidence_disp}_upper'] = cps_mond_cp_upper
        # when irradiance is less than self.irradiance_th_night, set the predicted value to 0
        self.data_part2.loc[(self.data_part2['irradiance'] < self.irradiance_th_night),
                            f'cps_mond_ci{confidence_disp}_lower'] = 0
        self.data_part2.loc[(self.data_part2['irradiance'] < self.irradiance_th_night),
                            f'cps_mond_ci{confidence_disp}_upper'] = 0

    def cp(self, confidence):
        lgb_cp = WrapRegressor(self.learner_prop)
        lgb_cp.calibrate(self.X_cal, self.y_cal, cps=False)
        # save the prediction interval with the following format:
        # cp_cp{confidence}_lower, cp_cp{confidence}_upper
        cp_cp_int = lgb_cp.predict_int(
            self.data_part2[self.list_of_features], y_min=0, y_max=1, confidence=confidence)
        cp_cp_lower = cp_cp_int[:, 0]
        cp_cp_upper = cp_cp_int[:, 1]
        # add the probabilities to the dataframe
        confidence_disp = int(confidence*100)
        self.data_part2[f'cp_ci{confidence_disp}_lower'] = cp_cp_lower
        self.data_part2[f'cp_ci{confidence_disp}_upper'] = cp_cp_upper
        # when irradiance is less than self.irradiance_th_night, set the predicted value to 0
        self.data_part2.loc[(self.data_part2['irradiance'] < self.irradiance_th_night),
                            f'cp_ci{confidence_disp}_lower'] = 0
        self.data_part2.loc[(self.data_part2['irradiance'] < self.irradiance_th_night),
                            f'cp_ci{confidence_disp}_upper'] = 0

    def plot_prob_reg(self, df, y, doy_start=0, doy_end=366, year=[2017, 2018, 2019], xlabel='Datetime', ylabel="example ylabel", layout=(7, 4), subplots=True, legends=[], figsize=(8, 4), fname='tmp_prob_plot.pdf'):
        # Filter data
        filtered_df = df[(df['doy'] >= doy_start) & (
            df['doy'] <= doy_end) & (df['year'].isin(year))]
        # convert datetime to datetime object
        filtered_df.loc[:, 'datetime'] = pd.to_datetime(
            filtered_df['datetime'])

        fig, ax = plt.subplots(figsize=figsize)
        # get the legends
        legends = self.legends
        # plot the first column as a solid line
        ax.plot(filtered_df['datetime'],
                filtered_df[y[0]], label=legends[0], color='black', linewidth=2)
        # plot the second column as a dashed line
        ax.plot(filtered_df['datetime'], filtered_df[y[1]],
                label=legends[1], color='red', linewidth=2, linestyle='--')
        # plot the rest of columns by pair, fill between each pair
        # the latter the pair, the lighter the color
        colors = sns.color_palette(
            "Reds",  n_colors=len(y)//2-1)[::-1]

        for i in range(2, len(y), 2):
            filtered_df.loc[:, y[i]] = pd.to_numeric(
                filtered_df[y[i]], errors='coerce')
            filtered_df.loc[:, y[i+1]] = pd.to_numeric(
                filtered_df[y[i+1]], errors='coerce')
            # Only add label for the first fill_between legends[2] and the last one legends[3]
            if i == 2:
                label = legends[2]
            elif i == len(y)-2:
                label = legends[3]
            else:
                label = None
            ax.fill_between(
                filtered_df['datetime'], filtered_df[y[i]], filtered_df[y[i+1]], alpha=0.4, color=colors[i//2-1], edgecolor='#b7b7b7', linewidth=0.5, label=label)
        # '#b7b7b7' is very light gray
        ax.legend()
        # format yyyy-mm-dd
        ax.xaxis.set_major_formatter(DateFormatter('%d-%H'))
        # Add major ticks at midnight (00:00) for each day
        ax.xaxis.set_major_locator(DayLocator())
        # add xlabel
        ax.set_xlabel(xlabel)
        # add ylabel
        ax.set_ylabel(ylabel)
        # save the plot
        plt.savefig(fname, format='pdf')

    def get_cols_to_plot_cp(self):
        # columns that has the format cp_ci{confidence}_lower
        cp_cols = [col for col in self.data_part2.columns if 'cp_ci' in col]
        self.cols_to_plot_cp = ['total_pv_gen_normalized',
                                'total_pv_gen_normalized_pred_lgb'] + cp_cols
        return self.cols_to_plot_cp

    def get_cols_to_plot_cps_mond(self):
        # columns that has the format cps_mond_cp10_lower
        cps_mond_cols = [
            col for col in self.data_part2.columns if 'cps_mond_ci' in col]
        self.cols_to_plot_cps_mond = ['total_pv_gen_normalized',
                                      'total_pv_gen_normalized_pred_lgb'] + cps_mond_cols
        return self.cols_to_plot_cps_mond

    def get_cols_to_plot_cps_mond_knn(self):
        # columns that has the format cps_mond_knn_ci{confidence}_lower
        cps_mond_knn_cols = [
            col for col in self.data_part2.columns if 'cps_mond_knn_ci' in col]
        self.cols_to_plot_cps_mond_knn = ['total_pv_gen_normalized',
                                          'total_pv_gen_normalized_pred_lgb'] + cps_mond_knn_cols
        return self.cols_to_plot_cps_mond_knn

    def get_cols_to_plot(self, method_name):
        if method_name == 'cps_mond_knn':
            cols_to_plot = self.get_cols_to_plot_cps_mond_knn()
        elif method_name == 'cps_mond':
            cols_to_plot = self.get_cols_to_plot_cps_mond()
        elif method_name == 'cp':
            cols_to_plot = self.get_cols_to_plot_cp()
        return cols_to_plot

    def plot_prob_reg_2019_Mar_1_to_10(self,  method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/../figs/prob_reg_{method_name}_2019_Mar_1_to_10.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=60, doy_end=68, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_1_to_4(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/../figs/prob_reg_{method_name}_2019_Mar_1_to_4.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=60, doy_end=62, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_1_to_2(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/prob_reg_{method_name}_2019_Mar_1_to_2.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=60, doy_end=60, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_2_to_3(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/prob_reg_{method_name}_2019_Mar_2_to_3.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=61, doy_end=61, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_3_to_4(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/prob_reg_{method_name}_2019_Mar_3_to_4.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=62, doy_end=62, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_4_to_5(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/prob_reg_{method_name}_2019_Mar_4_to_5.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=63, doy_end=63, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_5_to_6(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/prob_reg_{method_name}_2019_Mar_5_to_6.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=64, doy_end=64, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_6_to_7(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/prob_reg_{method_name}_2019_Mar_6_to_7.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=65, doy_end=65, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_7_to_8(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/prob_reg_{method_name}_2019_Mar_7_to_8.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=66, doy_end=66, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_8_to_9(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/prob_reg_{method_name}_2019_Mar_8_to_9.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=67, doy_end=67, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

    def plot_prob_reg_2019_Mar_9_to_10(self, method_name):
        cols_to_plot = self.get_cols_to_plot(method_name)
        fname = f'../figs/prob_reg_{method_name}_2019_Mar_9_to_10.pdf'
        self.plot_prob_reg(df=self.data_part2, y=cols_to_plot, doy_start=68, doy_end=68, year=[
            2019], xlabel='Datetime', ylabel="Normalized PV Generation", layout=(1, 1), subplots=False, figsize=(8, 4), fname=fname)

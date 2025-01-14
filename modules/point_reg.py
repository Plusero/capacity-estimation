import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import seaborn as sns
import scienceplots
from matplotlib.dates import DateFormatter, DayLocator
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


class point_regressor():
    def __init__(self, original_data: pd.DataFrame, list_of_features: list, y_col_name: str = 'total_pv_gen_normalized', random_seed: int = 42, irradiance_threshold: float = 0.01):
        self.list_of_features = list_of_features
        self.random_seed = random_seed
        self.data = original_data
        self.irradiance_threshold = irradiance_threshold
        self.selected_features = None
        self.y_pred_mlr = None
        self.y_pred_rfr = None
        self.y_pred_lgb = None
        self.res_mlr = None
        self.res_rfr = None
        self.res_lgb = None
        self.list_to_plot = ['total_pv_gen_normalized', 'total_pv_gen_normalized_predicted_mlr',
                             'total_pv_gen_normalized_predicted_rfr', 'total_pv_gen_normalized_predicted_lgb']
        self.y_col_name = y_col_name
        self.split_data()
        plt.style.use(['science'])

    def split_data(self):
        # It is essential to have exchangeability between calibration and test set
        # when Conformal Prediction is needed, use the following split
        #####################################
        # # 2017-04-02 to 2018-09-22 as train
        self.train_data = self.data[(self.data['datetime'] >= '2017-04-02')
                                    & (self.data['datetime'] <= '2018-09-22')]
        # 2018-09-23 to 2018-12-22 as calibration
        # autumn equinox to winter solstice
        self.cal_data = self.data[(self.data['datetime'] >= '2018-09-23') &
                                  (self.data['datetime'] <= '2018-12-22')]
        # 2018-12-23 to 2019-03-10 06:00:00 as test
        # winter solstice to spring equinox
        self.test_data = self.data[(self.data['datetime'] >= '2018-12-23')
                                   & (self.data['datetime'] <= '2019-03-10 06:00:00')]
        #####################################
        # when Conformal Prediction is not needed, use the following split
        #####################################
        # self.train_data = self.data[(self.data['datetime'] >= '2017-04-02')
        #                             & (self.data['datetime'] <= '2018-12-22')]
        # self.cal_data = self.data[(self.data['datetime'] >= '2018-09-23') &
        #                           (self.data['datetime'] <= '2018-12-22')]
        # self.test_data = self.data[(self.data['datetime'] >= '2018-12-23')
        #                            & (self.data['datetime'] <= '2019-03-10 06:00:00')]
        #####################################

        # Filter data where irradiance is greater than or equal to irradiance_threshold
        self.train_data = self.train_data[self.train_data['irradiance']
                                          >= self.irradiance_threshold]
        self.cal_data = self.cal_data[self.cal_data['irradiance']
                                      >= self.irradiance_threshold]
        self.test_data = self.test_data[self.test_data['irradiance']
                                        >= self.irradiance_threshold]

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

    def sequential_selection(self, direction: str = 'forward', n_features: int = 4):
        rfr = RandomForestRegressor(
            n_jobs=-1, n_estimators=128, random_state=self.random_seed)
        # Define the feature selector
        sfs = SequentialFeatureSelector(rfr,
                                        n_features_to_select=n_features,
                                        n_jobs=-1,
                                        direction=direction,
                                        cv=5,
                                        scoring='neg_root_mean_squared_error')
        sfs.fit(self.X_train, self.y_train)
        # print get support
        support_mask = sfs.get_support()
        print(f'{direction} selection: {support_mask}')
        # Convert list to numpy array for boolean indexing
        features_selected = np.array(self.list_of_features)[support_mask]
        print(
            f'{direction} selected features: {features_selected}')

        # if direction is forward,
        if direction == 'forward':
            self.forward_selected_features = features_selected
        # if direction is backward
        if direction == 'backward':
            self.backward_selected_features = features_selected
        # if both forward and backward selected features are the same
        if hasattr(self, 'forward_selected_features') and hasattr(self, 'backward_selected_features'):
            if np.array_equal(self.forward_selected_features, self.backward_selected_features):
                self.selected_features = self.forward_selected_features
        return None

    def sequential_selection_mlxtend(self, direction: str = 'forward', n_features: int = 4):
        rfr = RandomForestRegressor(
            n_jobs=-1, n_estimators=100, random_state=self.random_seed)

        # Define the feature selector
        sfs = SFS(rfr,
                  k_features=n_features,
                  n_jobs=-1,
                  forward=True if direction == 'forward' else False,
                  scoring='r2',
                  cv=5,
                  verbose=2)
        sfs.fit(self.X_train, self.y_train)
        # print get support
        support_mask = sfs.k_feature_idx_
        print(f'{direction} selection: {support_mask}')
        # use the support mask to get the features
        features_selected = [self.list_of_features[i] for i in support_mask]
        print(
            f'{direction} selected features: {features_selected}')

        # if direction is forward,
        if direction == 'forward':
            self.forward_selected_features = features_selected
        # if direction is backward
        if direction == 'backward':
            self.backward_selected_features = features_selected
        # if both forward and backward selected features are the same
        if hasattr(self, 'forward_selected_features') and hasattr(self, 'backward_selected_features'):
            if np.array_equal(self.forward_selected_features, self.backward_selected_features):
                self.selected_features = self.forward_selected_features

        print("\nAll Feature Subsets Evaluated:")
        for i, feature_subset in enumerate(sfs.subsets_.values(), start=1):
            print(f"Iteration {i}:")
            print(f"Selected Features: {feature_subset['feature_idx']}")
            print(f"Performance: {feature_subset['avg_score']}")
        # plot results of sfs
        fig = plot_sfs(sfs.get_metric_dict(),
                       kind='std_dev',
                       figsize=(6, 4),
                       ylabel='$\mathrm{R}^2$')
        plt.title(f'Sequential {direction.capitalize()} Selection (w. StdDev)')
        plt.grid()
        plt.savefig(
            f'../figs/sequential_{direction}_selection_w_stddev.pdf', format='pdf')
        plt.show()
        return None

    def remove_suffix_norm(self, labels: list):
        # remove suffix "_norm" from the label
        return [label.replace('_norm', '') for label in labels]

    def tree_based_selection_MDI(self):
        # mean decrease in impurity
        rf = self.rf_fitted_on_prop_train()
        importances = rf.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in rf.estimators_], axis=0)
        forest_importances = pd.Series(
            importances, index=self.list_of_features)
        # sort the forest_importances by the importances
        forest_importances = forest_importances.sort_values(ascending=True)

        # Set figure size
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot the horizontal bar chart
        # Use barh for horizontal bars
        forest_importances.plot.barh(xerr=std, ax=ax,  alpha=0.8, error_kw={
                                     'ecolor': 'black', 'capsize': 5, 'capthick': 1.5})
        # for the y-axis ticks, remove suffix of "_norm"
        # Get current tick labels
        labels = [label.get_text() for label in ax.get_yticklabels()]
        # Remove '_norm' from each label
        new_labels = self.remove_suffix_norm(labels)
        # Set the new labels
        ax.set_yticklabels(new_labels)

        # Set title and labels with increased font sizes
        ax.set_title("Feature importances using MDI",
                     fontsize=16)  # Title font size
        ax.set_xlabel("Mean decrease in impurity",
                      fontsize=14)  # X-axis label font size
        ax.set_ylabel("Features", fontsize=14)  # Y-axis label font size
        ax.tick_params(axis='x', labelsize=12)  # X-axis tick labels font size
        ax.tick_params(axis='y', labelsize=12)  # Y-axis tick labels font size

        fig.tight_layout()  # Adjust layout to avoid overlap
        fig.savefig(
            '../figs/feature_importance_MDI.pdf', format='pdf')

    def tree_based_selection_permutation(self):
        # Fit the random forest model on training data
        rf = self.rf_fitted_on_prop_train()

        # Perform permutation importance
        result = permutation_importance(
            estimator=rf,
            X=self.X_train,
            y=self.y_train,
            n_repeats=10,
            random_state=self.random_seed,
            n_jobs=-1
        )

        # Create a pandas Series for the importances
        forest_importances = pd.Series(
            result.importances_mean, index=self.list_of_features)
        # sort the forest_importances by the importances
        forest_importances = forest_importances.sort_values(ascending=True)

        # Set figure size
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot the horizontal bar chart
        forest_importances.plot.barh(
            xerr=result.importances_std, ax=ax,  alpha=0.8, error_kw={
                'ecolor': 'black', 'capsize': 5, 'capthick': 1.5}
        )
        # for the y-axis ticks, remove suffix of "_norm"
        # Get current tick labels
        labels = [label.get_text() for label in ax.get_yticklabels()]
        # Remove '_norm' from each label
        new_labels = self.remove_suffix_norm(labels)
        # Set the new labels
        ax.set_yticklabels(new_labels)
        # Set the title and labels with adjusted font sizes
        ax.set_title(
            "Feature importances using feature permutation", fontsize=16)
        # X-axis now represents the importance
        ax.set_xlabel("Mean accuracy decrease", fontsize=14)
        # Y-axis represents the feature names
        ax.set_ylabel("Features", fontsize=14)
        ax.tick_params(axis='x', labelsize=12)  # Adjust X-axis tick label size
        ax.tick_params(axis='y', labelsize=12)  # Adjust Y-axis tick label size

        # Adjust layout and show the plot
        fig.tight_layout()
        fig.savefig(
            '../figs/feature_importance_permutation.pdf', format='pdf')

    def rf_fitted_on_prop_train(self):
        rf = RandomForestRegressor(
            n_jobs=-1, n_estimators=100, random_state=self.random_seed)
        rf.fit(self.X_prop_train, self.y_prop_train)
        return rf

    def mlr_fitted_on_prop_train(self):
        mlr = LinearRegression()
        mlr.fit(self.X_prop_train, self.y_prop_train)
        return mlr

    def xgb_fitted_on_prop_train(self):
        xgb = XGBRegressor(random_state=self.random_seed)
        xgb.fit(self.X_prop_train, self.y_prop_train)
        return xgb

    def lightgbm_fitted_on_prop_train(self):
        lgb = LGBMRegressor(random_state=self.random_seed)
        lgb.fit(self.X_prop_train, self.y_prop_train)
        return lgb

    def add_trendline(self, x, y, ax):
        # Sort x and y values to ensure a continuous line
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Fit polynomial and create trendline
        z = np.polyfit(x_sorted, y_sorted, 5)
        p = np.poly1d(z)
        ax.plot(x_sorted, p(x_sorted), "--", color='red', linewidth=2)

    def compare_regressors(self, fname1: str = "regressor_comparison1.pdf", fname2: str = "regressor_comparison2.pdf"):
        self.use_selected_features()
        y_test = self.y_test
        # compare the regressors, MLR, RFR,LightGBM
        # 1. fit the regressors
        # 2. predict the test set
        # 3. compare the performance by R2, RMSE, actual vs predicted figure and histogram of residuals
        # fit the regressors
        rfr = self.rf_fitted_on_prop_train()
        mlr = self.mlr_fitted_on_prop_train()
        lgb = self.lightgbm_fitted_on_prop_train()
        # predict the test set
        y_pred_rfr = rfr.predict(self.X_test)
        y_pred_mlr = mlr.predict(self.X_test)
        y_pred_lgb = lgb.predict(self.X_test)
        # add to
        # calculate R2, RMSE
        r2_rfr, adj_r2_rfr = self.r2_score_actual_vs_predicted(
            y_test, y_pred_rfr)
        rmse_rfr = self.root_mean_squared_error(y_test, y_pred_rfr)
        r2_mlr, adj_r2_mlr = self.r2_score_actual_vs_predicted(
            y_test, y_pred_mlr)
        rmse_mlr = self.root_mean_squared_error(y_test, y_pred_mlr)
        r2_lgb, adj_r2_lgb = self.r2_score_actual_vs_predicted(
            y_test, y_pred_lgb)
        rmse_lgb = self.root_mean_squared_error(y_test, y_pred_lgb)
        res_rfr = y_test - y_pred_rfr
        res_mlr = y_test - y_pred_mlr
        res_lgb = y_test - y_pred_lgb
        # make a table to show the performance and save the table
        performance_table = pd.DataFrame({
            'R2': [r2_mlr, r2_rfr, r2_lgb],
            'Adjusted R2': [adj_r2_mlr, adj_r2_rfr, adj_r2_lgb],
            'RMSE': [rmse_mlr, rmse_rfr, rmse_lgb]
        })
        print(performance_table)
        performance_table.to_csv(
            '../figs/point_reg_performance_table.csv', index=False)
        # plot actual vs predicted
        # the first row are actual vs predicted plots, add a solid line of y=x
        fig1, axes1 = plt.subplots(1, 3, figsize=(
            # Define relative heights for the rows
            12, 4), sharey=True)
        axes1[0].scatter(x=y_test, y=y_pred_mlr, alpha=0.5)
        axes1[1].scatter(x=y_test, y=y_pred_rfr, alpha=0.5)
        axes1[2].scatter(x=y_test, y=y_pred_lgb, alpha=0.5)
        axes1[0].plot(y_test, y_test, 'k-', linewidth=2)
        self.add_trendline(y_test, y_pred_mlr, axes1[0])
        axes1[1].plot(y_test, y_test, 'k-', linewidth=2)
        self.add_trendline(y_test, y_pred_rfr, axes1[1])
        axes1[2].plot(y_test, y_test, 'k-', linewidth=2)
        self.add_trendline(y_test, y_pred_lgb, axes1[2])
        axes1[0].set_title('MLR')
        axes1[1].set_title('RFR')
        axes1[2].set_title('LGB')
        axes1[0].set_ylabel('Predicted values', fontsize=16)
        axes1[1].set_xlabel('Actual values', fontsize=16)
        axes1[0].tick_params(axis='x', labelsize=14)
        axes1[1].tick_params(axis='x', labelsize=14)
        axes1[2].tick_params(axis='x', labelsize=14)
        axes1[0].tick_params(axis='y', labelsize=14)
        axes1[1].tick_params(axis='y', labelsize=14)
        axes1[2].tick_params(axis='y', labelsize=14)
        axes1[0].set_aspect('equal')
        axes1[1].set_aspect('equal')
        axes1[2].set_aspect('equal')
        # save fig1
        fig1.savefig(
            f'../figs/{fname1}', format='pdf')
        fig2, axes2 = plt.subplots(1, 3, figsize=(
            12, 4), sharey=True, gridspec_kw={})
        # the second row are residual plots
        axes2[0].violinplot(res_mlr, positions=[-0.05], side='low')
        axes2[1].violinplot(res_rfr, positions=[-0.05], side='low')
        axes2[2].violinplot(res_lgb, positions=[-0.05], side='low')
        axes2[0].scatter(x=y_test, y=res_mlr, alpha=0.5)
        self.add_trendline(y_test, res_mlr, axes2[0])
        axes2[1].scatter(x=y_test, y=res_rfr, alpha=0.5)
        self.add_trendline(y_test, res_rfr, axes2[1])
        axes2[2].scatter(x=y_test, y=res_lgb, alpha=0.5)
        self.add_trendline(y_test, res_lgb, axes2[2])
        # add the solid black line of y=0
        axes2[0].plot(y_test, np.zeros(
            len(y_test)), 'k-', linewidth=2)
        axes2[1].plot(y_test, np.zeros(
            len(y_test)), 'k-', linewidth=2)
        axes2[2].plot(y_test, np.zeros(
            len(y_test)), 'k-', linewidth=2)
        axes2[0].set_title('MLR')
        axes2[1].set_title('RFR')
        axes2[2].set_title('LGB')
        axes2[0].set_ylabel('Residuals', fontsize=16)
        axes2[1].set_xlabel('Actual values', fontsize=16)
        axes2[0].tick_params(axis='x', labelsize=14)
        axes2[1].tick_params(axis='x', labelsize=14)
        axes2[2].tick_params(axis='x', labelsize=14)
        axes2[0].tick_params(axis='y', labelsize=14)
        axes2[1].tick_params(axis='y', labelsize=14)
        axes2[2].tick_params(axis='y', labelsize=14)
        fig2.tight_layout()
        axes2[1].set_xlabel('Actual values', fontsize=14)
        fig2.savefig(
            f'../figs/{fname2}', format='pdf')
        plt.show()
        # add results to the class
        self.y_pred_mlr = y_pred_mlr
        self.y_pred_rfr = y_pred_rfr
        self.y_pred_lgb = y_pred_lgb
        self.res_mlr = res_mlr
        self.res_rfr = res_rfr
        self.res_lgb = res_lgb
        # add full prediction results to the data
        self.data['total_pv_gen_normalized_predicted_mlr'] = mlr.predict(
            self.data[self.selected_features])
        self.data['total_pv_gen_normalized_predicted_rfr'] = rfr.predict(
            self.data[self.selected_features])
        self.data['total_pv_gen_normalized_predicted_lgb'] = lgb.predict(
            self.data[self.selected_features])
        # when irradiance is less than 0.01, set the predicted value to 0
        self.data.loc[(self.data['irradiance'] < 0.01),
                      'total_pv_gen_normalized_predicted_mlr'] = 0
        self.data.loc[(self.data['irradiance'] < 0.01),
                      'total_pv_gen_normalized_predicted_rfr'] = 0
        self.data.loc[(self.data['irradiance'] < 0.01),
                      'total_pv_gen_normalized_predicted_lgb'] = 0

    def compare_regressors_on_train(self, fname1: str = "regressor_comparison1.pdf", fname2: str = "regressor_comparison2.pdf"):
        self.use_selected_features()
        y_test = self.y_train
        print("shape of y_test", y_test.shape)
        rfr = self.rf_fitted_on_prop_train()
        mlr = self.mlr_fitted_on_prop_train()
        lgb = self.lightgbm_fitted_on_prop_train()
        # predict the test set
        y_pred_rfr = rfr.predict(self.X_train)
        y_pred_mlr = mlr.predict(self.X_train)
        y_pred_lgb = lgb.predict(self.X_train)
        print("shape of y_pred_rfr", y_pred_rfr.shape)
        # add to
        # calculate R2, RMSE
        r2_rfr, adj_r2_rfr = self.r2_score_actual_vs_predicted(
            y_test, y_pred_rfr)
        rmse_rfr = self.root_mean_squared_error(y_test, y_pred_rfr)
        r2_mlr, adj_r2_mlr = self.r2_score_actual_vs_predicted(
            y_test, y_pred_mlr)
        rmse_mlr = self.root_mean_squared_error(y_test, y_pred_mlr)
        r2_lgb, adj_r2_lgb = self.r2_score_actual_vs_predicted(
            y_test, y_pred_lgb)
        rmse_lgb = self.root_mean_squared_error(y_test, y_pred_lgb)
        res_rfr = y_test - y_pred_rfr
        res_mlr = y_test - y_pred_mlr
        res_lgb = y_test - y_pred_lgb
        # make a table to show the performance and save the table
        performance_table = pd.DataFrame({
            'R2': [r2_mlr, r2_rfr, r2_lgb],
            'Adjusted R2': [adj_r2_mlr, adj_r2_rfr, adj_r2_lgb],
            'RMSE': [rmse_mlr, rmse_rfr, rmse_lgb]
        })
        # plot actual vs predicted
        # the first row are actual vs predicted plots, add a solid line of y=x
        fig1, axes1 = plt.subplots(1, 3, figsize=(
            # Define relative heights for the rows
            12, 4), sharey=True)
        axes1[0].scatter(x=y_test, y=y_pred_mlr, alpha=0.5)
        axes1[1].scatter(x=y_test, y=y_pred_rfr, alpha=0.5)
        axes1[2].scatter(x=y_test, y=y_pred_lgb, alpha=0.5)
        axes1[0].plot(y_test, y_test, 'k-', linewidth=2)
        self.add_trendline(y_test, y_pred_mlr, axes1[0])
        axes1[1].plot(y_test, y_test, 'k-', linewidth=2)
        self.add_trendline(y_test, y_pred_rfr, axes1[1])
        axes1[2].plot(y_test, y_test, 'k-', linewidth=2)
        self.add_trendline(y_test, y_pred_lgb, axes1[2])
        axes1[0].set_title('MLR')
        axes1[1].set_title('RFR')
        axes1[2].set_title('LGB')
        axes1[0].set_ylabel('Predicted values', fontsize=16)
        axes1[1].set_xlabel('Actual values', fontsize=16)
        axes1[0].tick_params(axis='x', labelsize=14)
        axes1[1].tick_params(axis='x', labelsize=14)
        axes1[2].tick_params(axis='x', labelsize=14)
        axes1[0].tick_params(axis='y', labelsize=14)
        axes1[1].tick_params(axis='y', labelsize=14)
        axes1[2].tick_params(axis='y', labelsize=14)
        axes1[0].set_aspect('equal')
        axes1[1].set_aspect('equal')
        axes1[2].set_aspect('equal')
        # save fig1
        fig1.savefig(
            f'../figs/{fname1}', format='pdf')
        fig2, axes2 = plt.subplots(1, 3, figsize=(
            12, 4), sharey=True, gridspec_kw={})
        # the second row are residual plots
        axes2[0].violinplot(res_mlr, positions=[-0.05], side='low')
        axes2[1].violinplot(res_rfr, positions=[-0.05], side='low')
        axes2[2].violinplot(res_lgb, positions=[-0.05], side='low')
        axes2[0].scatter(x=y_test, y=res_mlr, alpha=0.5)
        self.add_trendline(y_test, res_mlr, axes2[0])
        axes2[1].scatter(x=y_test, y=res_rfr, alpha=0.5)
        self.add_trendline(y_test, res_rfr, axes2[1])
        axes2[2].scatter(x=y_test, y=res_lgb, alpha=0.5)
        self.add_trendline(y_test, res_lgb, axes2[2])
        # add the solid black line of y=0
        axes2[0].plot(y_test, np.zeros(
            len(y_test)), 'k-', linewidth=2)
        axes2[1].plot(y_test, np.zeros(
            len(y_test)), 'k-', linewidth=2)
        axes2[2].plot(y_test, np.zeros(
            len(y_test)), 'k-', linewidth=2)
        axes2[0].set_title('MLR')
        axes2[1].set_title('RFR')
        axes2[2].set_title('LGB')
        axes2[0].set_ylabel('Residuals', fontsize=16)
        axes2[1].set_xlabel('Actual values', fontsize=16)
        axes2[0].tick_params(axis='x', labelsize=14)
        axes2[1].tick_params(axis='x', labelsize=14)
        axes2[2].tick_params(axis='x', labelsize=14)
        axes2[0].tick_params(axis='y', labelsize=14)
        axes2[1].tick_params(axis='y', labelsize=14)
        axes2[2].tick_params(axis='y', labelsize=14)
        fig2.tight_layout()
        axes2[1].set_xlabel('Actual values', fontsize=14)
        fig2.savefig(
            f'../figs/{fname2}', format='pdf')
        plt.show()

    def r2_score_actual_vs_predicted(self, y_test, y_pred) -> None:
        r2 = r2_score(y_test, y_pred)
        n = len(y_test)
        p = self.X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return r2, adj_r2

    def root_mean_squared_error(self, y_test, y_pred) -> None:
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        return rmse

    def df_common_xylabel_plot(self, df, y, doy_start=0, doy_end=366, year=[2017, 2018, 2019], xlabel='Datetime', ylabel="example ylabel", layout=(7, 4), subplots=True, figsize=(8, 4), legends=[], fname=None):
        # Filter data
        filtered_df = df[(df['doy'] >= doy_start) & (
            df['doy'] <= doy_end) & (df['year'].isin(year))]
        # convert datetime to datetime object
        filtered_df.loc[:, 'datetime'] = pd.to_datetime(
            filtered_df['datetime'])

        if subplots:
            fig, axes = plt.subplots(
                layout[0], layout[1], figsize=figsize, sharex=True)
            axes = axes.flatten()

            for idx, col in enumerate(y):
                ax = axes[idx]
                ax.xaxis.axis_date('UTC')
                ax.plot(filtered_df['datetime'], filtered_df[col])
                ax.set_title(col)
                ax.xaxis.set_major_formatter(DateFormatter('%d-%H'))
                # Add major ticks at midnight (00:00) for each day
                ax.xaxis.set_major_locator(DayLocator())
                ax.tick_params(axis='both', which='major')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            for col in y:
                ax.xaxis.axis_date('UTC')
                ax.plot(filtered_df['datetime'], filtered_df[col], label=col)
            if legends:
                ax.legend(legends)
            else:
                ax.legend()
            # format yyyy-mm-dd
            ax.xaxis.set_major_formatter(DateFormatter('%d-%H'))
            # Add major ticks at midnight (00:00) for each day
            ax.xaxis.set_major_locator(DayLocator())
            ax.tick_params(axis='both', which='major')
        # Add common labels
        fig.text(0.05, 0.5, ylabel, va='center', rotation='vertical')
        fig.text(0.5, 0.0, xlabel, ha='center')

        # save the plot
        if fname:
            fig.savefig(
                f'../figs/{fname}', format='pdf')

    def visualize_results_by_dates_2018_Jan_7days(self, legends=[], fname="point_reg_2018_Jan_7days.pdf"):
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2018], doy_start=1, doy_end=7, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)

    def visualize_results_by_dates_2018_jul_7days(self, legends=[], fname="point_reg_2018_Jul_7days.pdf"):
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2018], doy_start=185, doy_end=191, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)

    def visualize_results_by_dates_2018_jul_2days(self, legends=[], fname="point_reg_2018_Jul_2days.pdf"):
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2018], doy_start=186, doy_end=187, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)

    def visualize_results_by_dates_2018_sep_7days(self, legends=[], fname="point_reg_2018_Sep_7days.pdf"):
        # plot 2018-09-23 to 2018-09-30
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2018], doy_start=274, doy_end=280, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)

    def visualize_results_by_dates_2018_sep_2days(self, legends=[], fname="point_reg_2018_Sep_2days.pdf"):
        # plot 2018-09-26 to 2018-09-28
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2018], doy_start=277, doy_end=278, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)

    def visualize_results_by_dates_2019_jan_7days(self, legends=[], fname="point_reg_2019_Jan_7days.pdf"):
        # plot 2019-01-01 to 2019-01-7
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2019], doy_start=1, doy_end=7, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)

    def visualize_results_by_dates_2019_Mar_7days(self, legends=[], fname="point_reg_2019_Mar_7days.pdf"):
        # plot 2019-03-01 to 2019-03-07
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2019], doy_start=60, doy_end=66, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)

    def visualize_results_by_dates_2019_Mar_2days(self, legends=[], fname="point_reg_2019_Mar_2days.pdf"):
        # plot 2019-03-05 to 2019-03-07
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2019], doy_start=64, doy_end=65, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)

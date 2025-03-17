import pandas as pd
import numpy as np
import pvlib
from sklearn.preprocessing import MinMaxScaler


class data_preprocessor():
    def __init__(self):
        self.pv_data = None
        self.pv_data_normalized = None
        self.con_data = None
        self.net_data = None
        self.ams_data = None
        self.ams_data_pv_normalized = None
        self.pv_capacity_from_pv_profiles = None
        self.pv_capacity_from_net = None
        # parameters for capacity estimation from net profiles
        self.irradiance_threshold_noon = 800
        self.irradiance_threshold_night = 0.01
        self.max_irradiance = 1000
        self.base_load = None
        self.correction_factors = None
        self.latitude = 52.317
        self.longitude = 4.79

    def estimate_pv_capacity_from_pv_profiles_avg_high_values(self):
        # drop the columns of time, weather, and weather normalized
        time_columns = ['datetime', 'Timestamp',
                        'HoD', 'dow', 'doy', 'month', 'year']
        weather_columns = ['temperature',
                           'precipitation', 'wind_speed', 'cloud_cover']
        weather_norm_columns = ['irradiance_norm', 'temperature_norm',
                                'precipitation_norm', 'wind_speed_norm', 'cloud_cover_norm']
        # Only drop columns that exist in the DataFrame
        columns_to_drop = [col for col in time_columns + weather_columns +
                           weather_norm_columns if col in self.pv_data.columns]
        df = self.pv_data.drop(columns=columns_to_drop)
        # can not be too high, otherwise no data sample for some households.
        irradiance_th = 500
        irradiance_max = 1000
        # get the rows where the irradiance is greater than the threshold
        pv_gen_high_irradiance = df[df['irradiance'] > irradiance_th]
        # rescale the pv_gen_high_irradiance by 1000/irradiance
        pv_gen_high_irradiance = pv_gen_high_irradiance.multiply(
            irradiance_max/pv_gen_high_irradiance['irradiance'], axis=0)
        # remove the irradiance column
        pv_gen_high_irradiance = pv_gen_high_irradiance.drop(
            columns=['irradiance'])
        # take the mean of the rescaled pv_gen_high_irradiance
        self.pv_capacity_from_pv_profiles = pv_gen_high_irradiance.mean()

        return None

    def drop_outliers_no_suffix(self, df: pd.DataFrame) -> None:
        # drop the columns of 9506H, because there is no solar panels.
        # drop the columns of 4226R, because there is weird data in 2019 March.
        # drop the column of 7743P, for the high missing rate of 0.948231.
        # drop the column of 7099P because none of its data is in summer and high missing rate 75.56%.
        # drop the columns of 3307S and 2108P, because they seem to have some demand response strategies.
        df = df.drop(columns=['9506H', '4226R', '7743P',
                     '7099P', '3307S', '2108P'])
        return df

    def normalize_pv(self) -> None:
        # normalize the PV generation of each household by the its installed capacity
        # by dividing all the xxxxx_pv columns with its maximum value
        pv_columns = [
            col for col in self.ams_data.columns if col.endswith('_pv')]
        df_normalized = self.ams_data.copy()
        df_normalized[pv_columns] = df_normalized[pv_columns].div(
            df_normalized[pv_columns].max())
        self.ams_data_pv_normalized = df_normalized
        self.pv_normalized = True
        return df_normalized

    def drop_outliers(self) -> None:
        # drop the columns of 9506H, because there is no solar panels.
        # drop the columns of 4226R, because there is weird data in 2019 March.
        # drop the column of 7743P, for the high missing rate of 0.948231.
        # drop the column of 7099P because none of its data is in summer and high missing rate 75.56%.
        # drop the columns of 3307S and 2108P, because they seem to have some demand response strategies.
        self.ams_data = self.ams_data.drop(columns=['9506H_pv', '9506H_con', '9506H_net',
                                                    '4226R_pv', '4226R_con', '4226R_net',
                                                    '7743P_pv', '7743P_con', '7743P_net',
                                                    '7099P_pv', '7099P_con', '7099P_net',
                                                    '3307S_pv', '3307S_con', '3307S_net',
                                                    '2108P_pv', '2108P_con', '2108P_net'])
        self.outliers_dropped = True
        return None

    def add_total_net(self):
        net_columns = [
            col for col in self.ams_data.columns if col.endswith('_net')]
        self.ams_data['total_net'] = self.ams_data[net_columns].sum(axis=1)
        self.ams_data_pv_normalized['total_net'] = self.ams_data['total_net']
        self.total_net_added = True
        return None

    def add_total_con(self):
        con_columns = [
            col for col in self.ams_data.columns if col.endswith('_con')]
        print(con_columns)
        self.ams_data['total_con'] = self.ams_data[con_columns].sum(axis=1)
        self.ams_data_pv_normalized['total_con'] = self.ams_data['total_con']
        self.total_con_added = True
        return None

    def add_total_pv_gen(self):
        # add a column of "total_pv_gen"
        # do not use the normalized pv data!
        pv_columns = [
            col for col in self.ams_data.columns if col.endswith('_pv')]
        print(pv_columns)
        self.ams_data['total_pv_gen'] = self.ams_data[pv_columns].sum(
            axis=1)
        self.ams_data_pv_normalized['total_pv_gen'] = self.ams_data['total_pv_gen']
        self.total_pv_gen_added = True
        return None

    def add_total_pv_gen_normalized(self, capacity: pd.Series):
        # default capacity is self.pv_capacity_from_pv_profiles
        # only when the household is generating pv power, its capacity is added to the total capacity.
        # For each row, the "normalized_total_pv_gen_correct" would be
        # (sum of pv generation)/(installed capacity of the households that has pv gen not NaN)
        df_tmp = self.ams_data.copy()
        # active_installed_capacity is the sum of installed capacity of the households that has pv gen not NaN
        # initialize active_installed_capacity, with the shape (len(df_tmp),)
        active_installed_capacity = np.zeros(len(df_tmp))
        # loop over each row to get the active_installed_capacity
        for index, row in df_tmp.iterrows():
            # check the pv_columns if the pv generation is not NaN, get the name of the columns
            pv_columns = [col for col in df_tmp.columns if col.endswith('_pv')]
            row_pv_columns = row[pv_columns]
            # for the row variable, get the index of those not NaN
            houses_not_nan = row_pv_columns[row_pv_columns.notna()].index
            # remove the "_pv" from the names in houses_not_nan
            houses_not_nan = [col.replace('_pv', '') for col in houses_not_nan]
            # Use houses_not_nan to get the installed capacity from df_installed_capacity
            active_installed_capacity[index] = capacity.loc[houses_not_nan].sum(
            )
        # add a column of "total_pv_gen_normalized"
        df_tmp['total_pv_gen_normalized'] = df_tmp['total_pv_gen'] / \
            active_installed_capacity
        self.ams_data_pv_normalized['total_pv_gen_normalized'] = df_tmp['total_pv_gen_normalized']
        self.total_pv_gen_normalized_added = True
        return None

    def estimate_pv_capacity_from_net_profiles(self):
        self.base_load_estimation()
        self.noon_load_estimation()
        # first use base_load - peak_load, then use the correction factor
        pv_gen_not_corrected = self.base_load - \
            self.high_irradiance_df[self.cols]
        pv_gen_corrected = pv_gen_not_corrected.multiply(
            self.correction_factors, axis=0)
        self.pv_capacity_from_net = pv_gen_corrected.mean()
        return None

    def base_load_estimation(self):
        # use the net load at night time as the base load
        # when irradiance is below threshold, the net load is the base load
        self.base_load = self.net_data[self.net_data['irradiance'] <
                                       self.irradiance_threshold_night][self.cols].mean()
        return None

    def noon_load_estimation(self):
        # when irradiance is above threshold, the net load is the noon load
        # add a correction factor here, which is max_irradiance/irradiance
        self.high_irradiance_df = self.net_data[self.net_data['irradiance']
                                                > self.irradiance_threshold_noon].copy()
        self.correction_factors = self.max_irradiance / \
            self.high_irradiance_df['irradiance']
        return None

############## Feature Engineering ##############
    def add_cos_sin_HoD(self):
        self.ams_data_pv_normalized['cos_HoD'] = np.cos(
            2*np.pi*self.ams_data_pv_normalized['HoD']/24)
        self.ams_data_pv_normalized['sin_HoD'] = np.sin(
            2*np.pi*self.ams_data_pv_normalized['HoD']/24)
        return None

    def add_solar_position_features(self):
        solarposition = pvlib.solarposition.get_solarposition(
            time=self.ams_data_pv_normalized['datetime'], latitude=self.latitude, longitude=self.longitude, temperature=self.ams_data_pv_normalized['temperature'])
        self.ams_data_pv_normalized['zenith'] = solarposition['apparent_zenith'].values
        self.ams_data_pv_normalized['azimuth'] = solarposition['azimuth'].values
        return None
############## Normalization ##############

    def normalize_feature(self, feature_list: list):
        scaler = MinMaxScaler()
        feature_list_norm = [col + '_norm' for col in feature_list]
        self.ams_data_pv_normalized[feature_list_norm] = scaler.fit_transform(
            self.ams_data_pv_normalized[feature_list])
        return None

############## Removal of night values ##############
    def remove_night_values_by_irradiance(self):
        self.ams_data_pv_normalized_night_removed_by_irradiance = self.ams_data_pv_normalized[
            self.ams_data_pv_normalized['irradiance'] > self.irradiance_threshold_night]
        return None

    def remove_night_values_by_hour_of_day(self):
        self.ams_data_pv_normalized_night_removed_by_hour_of_day = self.ams_data_pv_normalized[
            self.ams_data_pv_normalized['HoD'] > 6]
        return None

    def remove_night_values_by_azimuth(self):
        # remove the values where azimuth is outside of [90,270]
        self.ams_data_pv_normalized_night_removed_by_azimuth = self.ams_data_pv_normalized[
            (self.ams_data_pv_normalized['azimuth'] > 90) & (self.ams_data_pv_normalized['azimuth'] < 270)]
        return None

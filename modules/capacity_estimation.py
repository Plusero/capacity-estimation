import pandas as pd


class capacity_estimation_base_load:
    """
    A class for estimating the capacity of PV systems.
    Step 1: Sample the net load at night time as base load, when irradiance is low and PV gen is (nearly) zero.
    Step 2: Sample the load when the irradiance is below a certain threshold as noon load.
    Step 3: Calculate the capacity, which is (base load - noon load) * correction factor,
    where the correction factor is (max_irradiance/sample_irradiance). This correction factor originates the (nearly) linear relationship between irradiance and PV gen.
    """

    def __init__(self, df: pd.DataFrame, cols: list, irradiance_threshold_noon: float = 500, irradiance_threshold_night: float = 0.01, max_irradiance: float = 1000, base_load_correction_factor: float = None):
        self.df = df
        self.cols = cols
        self.irradiance_threshold_noon = irradiance_threshold_noon
        self.irradiance_threshold_night = irradiance_threshold_night
        self.max_irradiance = max_irradiance
        self.base_load_correction_factor = base_load_correction_factor
        # why higher threshold_noon?
        # because the when PV gen is dominating, the error is smaller.
        self.base_load = None
        self.correction_factors = None

    def estimate_capacity(self) -> pd.Series:
        self.base_load_estimation()
        self.noon_load_estimation()
        # first use base_load - peak_load, then use the correction factor
        pv_gen_not_corrected = self.base_load - \
            self.high_irradiance_df[self.cols]
        pv_gen_corrected = pv_gen_not_corrected.multiply(
            self.correction_factors, axis=0)
        pv_gen_capacity = pv_gen_corrected.mean()
        return pv_gen_capacity

    def base_load_estimation(self):
        # use the net load at night time as the base load
        # when irradiance is below threshold, the net load is the base load
        self.base_load_at_night = self.df[self.df['irradiance']
                                          < self.irradiance_threshold_night][self.cols].mean()
        # calculate the correction factor based on total_con(at daylight)/total_con(at night)
        if self.base_load_correction_factor is None:
            self.base_load_correction_factor = self.df[self.df['irradiance'] > self.irradiance_threshold_noon]['total_con'].mean(
            ) / self.df[self.df['irradiance'] <= self.irradiance_threshold_night]['total_con'].mean()
        self.base_load = self.base_load_at_night * self.base_load_correction_factor
        return None

    def noon_load_estimation(self):
        # add a correction factor here, which is max_irradiance/irradiance
        self.high_irradiance_df = self.df[self.df['irradiance']
                                          > self.irradiance_threshold_noon].copy()
        self.correction_factors = self.max_irradiance / \
            self.high_irradiance_df['irradiance']
        return None

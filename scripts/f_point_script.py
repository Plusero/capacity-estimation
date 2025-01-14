import sys  # nopep8
import os  # nopep8
sys.path.append(os.path.abspath('../scripts'))  # nopep8


from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from f_common_script import *

# list_of_features = ['irradiance', 'wind_speed', 'temperature',
#                     'precipitation', 'cloud_cover', 'cos_HoD', 'total_net', 'zenith', 'azimuth']
list_of_features = ['irradiance_norm', 'wind_speed_norm', 'temperature_norm',
                    'precipitation_norm', 'cloud_cover_norm', 'cos_HoD', 'total_net_norm', 'zenith_norm', 'azimuth_norm']
# Section:Functions for data processing


def add_suffix_to_pv_household(target: str) -> str:
    # add suffix '_pv' to the target if it matches the pattern of pv household
    if re.match(r'^\d{4}[a-zA-Z]$', target):
        return target+'_pv'
    else:
        return target


# Section: Functions for point regression


def load_split_predict(df: pd.DataFrame, list_of_features: list[str], target: str, train_size: float = 0.5, test_size: float = 0.25, shuffle: bool = False, model: str = 'rfr') -> None:
    # get the data for the specific household
    X = df[list_of_features]
    # get the target
    target_ready = add_suffix_to_pv_household(target)
    y = df[target_ready]
    data = remove_nan_rows(X, y)
    X = data[list_of_features]
    y = data[target_ready]
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=my_seed, shuffle=shuffle)
    if model == 'rfr':
        regressor = rfr_train(X_train, y_train)
    elif model == 'xgb':
        regressor = xgb_train(X_train, y_train)
    elif model == 'mlr':
        regressor = mlr_train(X_train, y_train)
    # make predictions
    y_pred = regressor.predict(X_test)
    r2_score_actual_vs_predicted(X_test, y_test, y_pred)
    root_mean_squared_error(y_test, y_pred)
    plot_actual_vs_predicted(y_test, y_pred)
    return regressor


def load_split_predict_without_night(df: pd.DataFrame, list_of_features: list[str], target: str, train_size: float = 0.5, test_size: float = 0.25, shuffle: bool = False, model: str = 'rfr') -> None:
    # get the data for the specific household
    X = df[list_of_features]
    target_ready = add_suffix_to_pv_household(target)
    y = df[target_ready]
    data = remove_nan_rows(X, y)
    # remove the night values
    data = remove_night_values(data)
    X = data[list_of_features]
    y = data[target_ready]
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=my_seed, shuffle=shuffle)
    if model == 'rfr':
        regressor = rfr_train(X_train, y_train)
    elif model == 'xgb':
        regressor = xgb_train(X_train, y_train)
    elif model == 'mlr':
        regressor = mlr_train(X_train, y_train)
    # make predictions
    _, X_test, _, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=my_seed, shuffle=shuffle)
    y_pred = regressor.predict(X_test)
    r2_score_actual_vs_predicted(X_test, y_test, y_pred)
    root_mean_squared_error(y_test, y_pred)
    plot_actual_vs_predicted(y_test, y_pred)
    return regressor


def remove_night_values(df) -> pd.DataFrame:
    # if irradiance exists, remove the rows where irradiance < 0.1
    if 'irradiance' in df.columns:
        df = df[df['irradiance'] >= 0.1]
    # if irradiance_norm exists, remove the rows where irradiance_norm < 0.001
    if 'irradiance_norm' in df.columns:
        df = df[df['irradiance_norm'] >= 0.001]
    return df


def remove_nan_rows(X, y) -> pd.DataFrame:
    data = pd.concat([X, y], axis=1)
    data = data.dropna()
    return data


def rfr_train(X_train, y_train) -> RandomForestRegressor:
    rfr = RandomForestRegressor(n_estimators=100, random_state=my_seed)
    rfr.fit(X_train, y_train)
    return rfr


def xgb_train(X_train, y_train) -> XGBRegressor:
    xgb = XGBRegressor(objective='reg:squarederror',
                       n_estimators=100, random_state=my_seed)
    xgb.fit(X_train, y_train)
    return xgb


def mlr_train(X_train, y_train) -> LinearRegression:
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    return mlr


# Section:Functions for evaluation of point regression


def r2_score_actual_vs_predicted(X_test, y_test, y_pred) -> None:
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r2}")
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"Adjusted R^2 Score: {adj_r2}")


def root_mean_squared_error(y_test, y_pred) -> None:
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"Root Mean Squared Error: {rmse}")

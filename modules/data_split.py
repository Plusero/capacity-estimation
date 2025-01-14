import matplotlib.pyplot as plt
import numpy as np


def split_data(data, list_of_features):
    # It is essential to have exchangeability between calibration and test set
    # 2017-04-02 to 2018-09-22 as train
    train_data = data[(data['datetime'] >= '2017-04-02')
                      & (data['datetime'] <= '2018-09-22')]
    # 2018-09-23 to 2018-12-22 as calibration
    # autumn equinox to winter solstice
    cal_data = data[(data['datetime'] >= '2018-09-23') &
                    (data['datetime'] <= '2018-12-22')]
    # 2018-12-23 to 2019-03-10 06:00:00 as test
    # winter solstice to spring equinox
    test_data = data[(data['datetime'] >= '2018-12-23')
                     & (data['datetime'] <= '2019-03-10 06:00:00')]
    print(train_data.columns)
    X_prop_train = train_data[list_of_features]
    X_cal = cal_data[list_of_features]
    X_test = test_data[list_of_features]
    X_train = np.concatenate([X_prop_train, X_cal], axis=0)
    y_prop_train = train_data['total_pv_gen_normalized']
    y_cal = cal_data['total_pv_gen_normalized']
    y_test = test_data['total_pv_gen_normalized']
    y_train = np.concatenate([y_prop_train, y_cal], axis=0)
    # convert y_test to numpy array
    y_test = np.array(y_test)
    # check shapes
    print(
        f'X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}, X_prop_train.shape: {X_prop_train.shape}, X_cal.shape: {X_cal.shape}')
    # plot the histogram of irradiance of X_cal, add a title
    plt.hist(X_cal['irradiance_norm'], bins=100)
    plt.title('Histogram of irradiance of X_cal')
    plt.show()
    # plot the histogram of irradiance of X_test
    plt.hist(X_test['irradiance_norm'], bins=100)
    plt.title('Histogram of irradiance of X_test')
    plt.show()
    # plot the histogram of irradiance of X_prop_train
    plt.hist(X_prop_train['irradiance_norm'], bins=100)
    plt.title('Histogram of irradiance of X_prop_train')
    plt.show()

    return X_train, X_prop_train, X_cal, X_test, y_train, y_prop_train, y_cal, y_test

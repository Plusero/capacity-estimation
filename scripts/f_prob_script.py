# Section:Importing libraries
from crepes import WrapRegressor
from crepes.extras import margin, DifficultyEstimator, MondrianCategorizer
from mapie.regression import MapieQuantileRegressor
import lightgbm
from lightgbm import LGBMRegressor
import numpy as np
import scienceplots
# Section:global variables

my_confidence = 0.9
confidences = [i/100 for i in range(5, 100, 5)]
my_no_bins = 6
my_k = 12
my_seed = 42
# set the seed of numpy, so that the Mondrian Categorizer is reproducible
np.random.seed(my_seed)

# Section: Functions for Mondrian Categorizer


def get_values(X):
    # The function get_values(X) is returning X[:, 0], which takes only the first column of X. Ensure that:
    # X is indeed a NumPy array or some data structure that supports indexing with [:,0].
    # If X is not a NumPy array, this operation could raise an error.
    # convert X to numpy array
    X = np.array(X)
    # returns only the first column of X
    # where the first column represents a significant feature derived from the Laplace matrix.
    # Leading eigenvalue from Laplace matrix
    return X[:, 0]


# Section:Implementation of probabilistic metrics

def calc_penalty(y_test, lower, upper, confidence_i):
    # if the prediction is outside the interval, calculate the penalty
    penalty = 0
    alpha = 1-confidence_i
    for i in range(len(y_test)):
        if y_test[i] < lower[i]:
            penalty += 2*(lower[i]-y_test[i])/alpha
        elif y_test[i] > upper[i]:
            penalty += 2*(y_test[i]-upper[i])/alpha
    return penalty/len(y_test)


def calculate_coverage(y_test, intervals):
    # calculate the coverage
    return np.sum([1 if (y_test[i] >= intervals[i, 0] and
                         y_test[i] <= intervals[i, 1]) else 0
                   for i in range(len(y_test))])/len(y_test)


def calculate_mean_size(intervals):
    return (intervals[:, 1]-intervals[:, 0]).mean()


def calculate_median_size(intervals):
    return np.median((intervals[:, 1]-intervals[:, 0]))

# Section: Functions for multi-level evaluation


def calc_multi_level(learner, X_test, y_test):
    coverages = []
    mean_sizes = []
    penalties = []
    for confidence_i in confidences:
        cp_int = learner.predict_int(
            X_test, y_min=0, y_max=1, confidence=confidence_i, seed=my_seed)
        lower = cp_int[:, 0]
        upper = cp_int[:, 1]
        coverages.append(calculate_coverage(y_test, cp_int))
        mean_sizes.append(calculate_mean_size(cp_int))
        penalties.append(calc_penalty(y_test, lower, upper, confidence_i))
    sharpness = np.mean(mean_sizes)
    calibration_error = np.mean(penalties)
    interval_score = sharpness+calibration_error
    return sharpness, calibration_error, interval_score


def calc_multi_level_WIS(learner, X_test, y_test):
    K = len(confidences)
    interval_scores = []
    wk_interval_scores = np.zeros((len(y_test), len(confidences)))
    w0 = 1/2
    for idx, confidence_i in enumerate(confidences):
        cp_int = learner.predict_int(
            X_test, y_min=0, y_max=1, confidence=confidence_i, seed=my_seed)
        lower = cp_int[:, 0]
        upper = cp_int[:, 1]
        width = upper-lower
        alpha_i = 1-confidence_i
        # vectorized penalty calculation
        penalties = np.zeros_like(y_test)
        mask_lower = y_test < lower
        mask_upper = y_test > upper
        penalties[mask_lower] = 2 * \
            (lower[mask_lower] - y_test[mask_lower])/alpha_i
        penalties[mask_upper] = 2 * \
            (y_test[mask_upper] - upper[mask_upper])/alpha_i
        interval_score = width + penalties
        interval_scores.append(interval_score)
        wk_interval_scores[:, idx] = alpha_i/2 * interval_score

    interval_scores = np.array(interval_scores)
    # m is the predictive median
    # for all confidences, take the median of the prediction intervals
    # median in confidence dimension
    m = learner.predict(X_test)
    # vectorized weighted interval score calculation
    weighted_interval_scores = 1 / \
        (K+1/2) * (w0*np.abs(y_test-m) + np.sum(wk_interval_scores, axis=1))

    return np.mean(weighted_interval_scores)


def calc_multi_level_cqr(X_test, y_test, confidences, X_prop_train, y_prop_train, X_cal, y_cal, calibrate=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coverages = []
    mean_sizes = []
    penalties = []
    for confidence_i in confidences:
        alpha = 1-confidence_i
        # if calibrate is True, use X_cal and y_cal for calibration
        estimator = lightgbm.LGBMRegressor(
            objective='quantile', alpha=alpha, random_state=my_seed, verbose=-1)
        if calibrate:
            cqr_reg = MapieQuantileRegressor(
                estimator=estimator, alpha=alpha)
            cqr_reg.fit(X=X_prop_train, y=y_prop_train,
                        X_calib=X_cal, y_calib=y_cal, random_state=my_seed)
            cqr_pred, cqr_int = cqr_reg.predict(X_test, alpha=alpha)
        else:
            cqr_reg = MapieQuantileRegressor(
                estimator=estimator, alpha=alpha)
            cqr_reg.fit(X=X_prop_train, y=y_prop_train, random_state=my_seed)
            # when not calibrated with calibration set, do not specify alpha here.
            cqr_pred, cqr_int = cqr_reg.predict(X_test)
        # limit the prediction interval to the range [0,1]
        cqr_int = np.clip(cqr_int, 0, 1)
        lower = cqr_int[:, 0]
        upper = cqr_int[:, 1]
        coverages.append(calculate_coverage(y_test, cqr_int))
        mean_sizes.append(calculate_mean_size(cqr_int))
        penalties.append(calc_penalty(y_test, lower, upper, confidence_i))
    sharpness = np.mean(mean_sizes)
    calibration_error = np.mean(penalties)
    interval_score = sharpness+calibration_error
    return sharpness, calibration_error, interval_score


def calc_multi_level_cqr_WIS(X_test, y_test, confidences, X_prop_train, y_prop_train, X_cal, y_cal, calibrate=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = len(confidences)
    interval_scores = []
    wk_interval_scores = np.zeros((len(y_test), len(confidences)))
    w0 = 1/2
    for idx, confidence_i in enumerate(confidences):
        alpha = 1-confidence_i
        # if calibrate is True, use X_cal and y_cal for calibration
        estimator = lightgbm.LGBMRegressor(
            objective='quantile', alpha=alpha, random_state=my_seed, verbose=-1)
        if calibrate:
            cqr_reg = MapieQuantileRegressor(
                estimator=estimator, alpha=alpha)
            cqr_reg.fit(X=X_prop_train, y=y_prop_train,
                        X_calib=X_cal, y_calib=y_cal, random_state=my_seed)
            cqr_pred, cqr_int = cqr_reg.predict(X_test, alpha=alpha)
        else:
            cqr_reg = MapieQuantileRegressor(
                estimator=estimator, alpha=alpha)
            cqr_reg.fit(X=X_prop_train, y=y_prop_train, random_state=my_seed)
            cqr_pred, cqr_int = cqr_reg.predict(X_test)
        # limit the prediction interval to the range [0,1]
        cqr_int = np.clip(cqr_int, 0, 1)
        lower = cqr_int[:, 0]
        upper = cqr_int[:, 1]
        # reshape to match the shape of y_test. e.g. from (757, 1) to (757,)
        lower = lower.reshape(-1)
        upper = upper.reshape(-1)
        width = upper-lower
        alpha_i = 1-confidence_i
        # vectorized penalty calculation
        penalties = np.zeros_like(y_test)
        mask_lower = y_test < lower
        mask_upper = y_test > upper
        penalties[mask_lower] = 2 * \
            (lower[mask_lower] - y_test[mask_lower])/alpha_i
        penalties[mask_upper] = 2 * \
            (y_test[mask_upper] - upper[mask_upper])/alpha_i
        interval_score = width + penalties
        interval_scores.append(interval_score)
        wk_interval_scores[:, idx] = alpha_i/2 * interval_score
    interval_scores = np.array(interval_scores)
    # m is the predictive median
    # for all confidences, take the median of the prediction intervals
    # median in confidence dimension
    m = cqr_pred
    # vectorized weighted interval score calculation
    weighted_interval_scores = 1 / \
        (K+1/2) * (w0*np.abs(y_test-m) + np.sum(wk_interval_scores, axis=1))

    return np.mean(weighted_interval_scores)

import math
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample


def bootstrap_confidence_interval(truth: List[float], pred: List[float],
                                  stat_function: Callable[[List[float], List[float]], float],
                                  num_iterations: int = 1000, interval: float = 95.0) -> Tuple[float, float, float]:
    """ Calculate a 95% confidence interval (CI) for a statistic of interest using bootstrap

    :param truth: the true values
    :param pred: the predicted values
    :param stat_function: the function to calculate the statistic of interest, should return a single value
    :param num_iterations: number of bootstrap iterations
    :param interval: the confidence interval to calculate
    :return: 95% CI lower bound, value of the statistic, 95% CI upper bound
    """
    lb = (100.0 - interval) / 2.0
    ub = 100.0 - lb

    result_df = pd.DataFrame({"truth": truth, "pred": pred})
    stat_val = stat_function(truth, pred)
    stat_list = []
    for _ in range(0, num_iterations):
        sample_df = resample(result_df)
        stat_list.append(roc_auc_score(sample_df.truth, sample_df.pred))
    return np.percentile(stat_list, lb), stat_val, np.percentile(stat_list, ub)


def pearson_confidence(r: int, num: int, interval: float = 0.95) -> Tuple[float, float]:
    """
    Calculate upper and lower 95% CI for a Pearson r (not R**2)
    Inspired by https://stats.stackexchange.com/questions/18887

    :param r: Pearson's R
    :param num: number of data points
    :param interval: confidence interval (0-1.0)
    :return: lower bound, upper bound
    """
    stderr = 1.0 / math.sqrt(num - 3)
    z_score = norm.ppf(interval)
    delta = z_score * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)
    return lower, upper


def max_possible_correlation(vals: List[float], error: float = 1 / 3.0,
                             method: Callable[[List[float], List[float]], float] = pearsonr,
                             cycles: int = 1000) -> float:
    """
    Calculate the maximum possible correlation given a particular experimental error
    Based on Brown, Muchmore, Hajduk http://www.sciencedirect.com/science/article/pii/S1359644609000403
    :param vals: experimental values (should be on a log scale)
    :param error: experimental error
    :param method: method for calculating the correlation, must take 2 lists and return correlation and p_value
    :param cycles: number of random cycles
    :return: maximum possible correlation
    """
    cor_list = []
    for i in range(0, cycles):
        noisy_vals = []
        for val in vals:
            noisy_vals.append(val + np.random.normal(0, error))
        cor_list.append(method(vals, noisy_vals)[0])
    return np.mean(cor_list)

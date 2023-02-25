import numpy as np
from sklearn.metrics import mean_squared_error
def calc_stats_rmse(yts, yts_pred):
    diff = yts - yts_pred

    diff_mean = np.mean(diff)
    mse = (np.square(diff)).mean(axis=0)
    rmse = np.sqrt(mse)
    variance = 1/(len(diff)-1)*(np.square(diff-diff_mean)).sum(axis=0)
    std = np.sqrt(variance)  # Standard deviation
    sem = std/np.sqrt(len(diff))  # standard error of the mean

    # Normalize
    mean = np.mean(yts)
    nrmse = rmse/mean
    nstd = std/mean
    return nrmse, nstd

def calc_stats_over_all_param(yts, yts_pred, indices):
    diff = yts - yts_pred
    diff = diff[:, indices]

    diff_mean = np.mean(diff)
    mse = (np.square(diff)).mean()
    rmse = np.sqrt(mse)
    variance = 1/(len(diff)-1)*(np.square(diff-diff_mean)).sum()
    std = np.sqrt(variance)  # Standard deviation
    sem = std/np.sqrt(len(diff))  # standard error of the mean

    # Normalize
    mean = np.mean(yts)
    nrmse = rmse/mean
    nstd = std/mean
    return nrmse, nstd

def calc_stats(yts, yts_pred, truncate=True):
    diff = yts - yts_pred

    diff_mean = np.mean(diff)
    mse = (np.square(diff)).mean(axis=0)
    rmse = np.sqrt(mse)
    variance = 1/(len(diff)-1)*(np.square(diff-diff_mean)).sum(axis=0)
    std = np.sqrt(variance)  # Standard deviation
    sem = std/np.sqrt(len(diff))  # standard error of the mean

    # Normalize
    mean = np.mean(yts)
    nmse = mse/mean
    nstd = std/mean

    if not truncate:
        return nmse, nstd
    else:
        n, d = diff.shape

        trunc_diff_ll = estimate_quartiles(diff)

        tmse_l = []
        tstd_l = []
        for i in range(d):
            tmse = (np.square(trunc_diff_ll[i])).mean()
            tstd = np.std(trunc_diff_ll[i])
            tmse_l.append(tmse)
            tstd_l.append(tstd)
        tmse = np.array(tmse_l)
        tstd = np.array(tstd_l)
        tnmse = tmse/mean # Truncated (removed upper and lower quartiles normalized mse
        tnstd = tstd/mean # Truncated normalized std
        return nmse, nstd, tnmse, tnstd

def estimate_quartiles(diff):
    n, d = diff.shape

    q75, q25 = np.percentile(diff, [75, 25], axis=0)
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    max = max.reshape(1, d)
    min = min.reshape(1, d)

    idx_min = np.where(diff < min)
    idx_max = np.where(diff > max)
    diff[idx_min[0], idx_min[1]] = np.nan
    diff[idx_max[0], idx_max[1]] = np.nan

    #filter nans
    indices = ~np.isnan(diff)

    diff_l = []
    for i in range(d):
        diff_l.append(diff[indices[:, i], i])
    return diff_l
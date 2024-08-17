#
import random
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.feature_selection import mutual_info_classif, RFECV
from scipy.stats import spearmanr, kendalltau

#
from func import search_hb, test_eg
from data.data_load import load_coal_daily

#
data_coal = load_coal_daily()

data_coal_pct = data_coal.pct_change(fill_method=None).dropna()
# data_gold_pct = data_gold_pct.drop(columns=['Volume__PENOLES'])

"""
for c in data_coal.columns:
    data_coal[c] = data_coal[c] / data_coal[c].values[0]
for c in data_platinum.columns:
    data_platinum[c] = data_platinum[c] / data_platinum[c].values[0]
data_coal = data_coal.dropna()
data_platinum = data_platinum.dropna()
"""


alpha = 0.05
kwg = {'trend': 'c', 'max_lag': None, 'auto_lag': 'aic'}
resulted_coal = search_hb(data=data_coal_pct[['Close__COALCOMPANY', 'close__COAL']].values, test=test_eg, test_kwargs=kwg, alpha=alpha)
resulted_coal = {x: (data_coal_pct[['Close__COALCOMPANY', 'close__COAL']].columns.values[resulted_coal[x][0]],
                       data_coal_pct[['Close__COALCOMPANY', 'close__COAL']].columns.values[resulted_coal[x][1]]) for x in resulted_coal.keys()}

'''

def long_window_lag_pct(data, n_lags):
    data_w = data.copy()
    for c in data.columns:
        for j in range(n_lags):
            data_w['{0}_LW{1}'.format(c, j + 1)] = numpy.power(
                data_w[c].shift(periods=1) / data_w[c].shift(periods=j + 2), 1 / (j + 1)) - 1
    data_w = data_w.drop(columns=data.columns.values)
    return data_w


def lag(data, n_lags):
    data_lagged = data.copy()
    for j in range(n_lags):
        data_lagged[[x + '_LAG{0}'.format(j + 1) for x in data.columns.values]] = data.shift(periods=(j + 1))
    return data_lagged


def conseq_up(row):
    if pandas.isna(row).any():
        return numpy.nan
    else:
        return int(all([x > 0 for x in row]))


def conseq_dw(row):
    if pandas.isna(row).any():
        return numpy.nan
    else:
        return int(all([x > 0 for x in row]))


def conseq_check(data, max_lags):
    data_checked = data.copy()
    cols = [x for x in data.columns if 'LAG' not in x]
    for c in cols:
        for max_lag in max_lags:
            data_checked['{0}_QU{1}'.format(c, max_lag)] = data_checked[
                ['{0}_LAG{1}'.format(c, j + 1) for j in range(max_lag)]].apply(func=conseq_up, axis=1)
            data_checked['{0}_QD{1}'.format(c, max_lag)] = data_checked[
                ['{0}_LAG{1}'.format(c, j + 1) for j in range(max_lag)]].apply(func=conseq_dw, axis=1)
    data_checked = data_checked[[x for x in data_checked.columns if 'QU' in x or 'QD' in x]]
    return data_checked


def below_mean_conf(data, c, p):
    data_below = data.copy()
    for col in data_below.columns:
        series = data_below[col]
        belows = numpy.full(shape=(series.shape[0],), fill_value=pandas.NA)
        for t_low in range(series.shape[0] - p - 1):
            t_up = t_low + p
            sub = series[t_low:t_up]
            support = sub.mean() - (c * sub.std() * numpy.power(sub.shape[0], -0.5))
            belows[t_up + 1] = (series[t_up] <= support).astype(dtype=int)
        data_below['{0}_CFL{1}_{2}'.format(col, p, c)] = belows
    data_below = data_below.drop(columns=data.columns)
    return data_below


def beyond_mean_conf(data, c, p):
    data_beyond = data.copy()
    for col in data_beyond.columns:
        series = data_beyond[col]
        beyonds = numpy.full(shape=(series.shape[0],), fill_value=pandas.NA)
        for t_low in range(series.shape[0] - p - 1):
            t_up = t_low + p
            sub = series[t_low:t_up]
            support = sub.mean() + (c * sub.std() * numpy.power(sub.shape[0], -0.5))
            beyonds[t_up + 1] = (series[t_up] >= support).astype(dtype=int)
        data_beyond['{0}_CFY{1}_{2}'.format(col, p, c)] = beyonds
    data_beyond = data_beyond.drop(columns=data.columns)
    return data_beyond


data_coal_conf = pandas.concat((below_mean_conf(data=data_coal_pct, c=0.95, p=5),
                                below_mean_conf(data=data_coal_pct, c=0.95, p=10),
                                below_mean_conf(data=data_coal_pct, c=0.95, p=20),
                                beyond_mean_conf(data=data_coal_pct, c=0.95, p=5),
                                beyond_mean_conf(data=data_coal_pct, c=0.95, p=10),
                                beyond_mean_conf(data=data_coal_pct, c=0.95, p=20)), axis=1)

data_coal_mean = pandas.concat((data_coal_pct.shift(periods=1).rolling(window=5).mean().rename(
    columns={x: '{0}__MW5'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=10).mean().rename(
                                    columns={x: '{0}__MW10'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=20).mean().rename(
                                    columns={x: '{0}__MW20'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=5).mean().rename(
                                    columns={x: '{0}__MN5'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=10).mean().rename(
                                    columns={x: '{0}__MN10'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=20).mean().rename(
                                    columns={x: '{0}__MN20'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=5).mean().rename(
                                    columns={x: '{0}__MX5'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=10).mean().rename(
                                    columns={x: '{0}__MX10'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=20).mean().rename(
                                    columns={x: '{0}__MX20'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=5).mean().rename(
                                    columns={x: '{0}__SD5'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=10).mean().rename(
                                    columns={x: '{0}__SD10'.format(x) for x in data_coal_pct.columns}),
                                data_coal_pct.shift(periods=1).rolling(window=20).mean().rename(
                                    columns={x: '{0}__SD20'.format(x) for x in data_coal_pct.columns}),
                                ), axis=1)
data_coal_lw = long_window_lag_pct(data=data_coal, n_lags=20)
data_coal_lagged_pct = lag(data_coal_pct, n_lags=20)
data_coal_conseq = conseq_check(data=data_coal_lagged_pct, max_lags=[5, 10, 20])

data_joint = data_coal_lagged_pct.copy()
# data_joint = data_joint.merge(right=data_coal_lw, left_index=True, right_index=True, how='inner')
# data_joint = data_joint.merge(right=data_coal_mean, left_index=True, right_index=True, how='inner')
# data_joint = data_joint.merge(right=data_coal_conseq, left_index=True, right_index=True, how='inner')
# data_joint = data_joint.merge(right=data_coal_conf, left_index=True, right_index=True, how='inner')
data_joint = data_joint.dropna()

x_factors = [x for x in data_joint.columns if
             any([y in x for y in ['LAG', 'LW', 'QU', 'QD', 'CFL', 'CFY', 'MW', 'MN', 'MX', 'SD']])]
X = data_joint[x_factors].values
Y = ((data_joint['open__BHP'] - data_joint['value__COAL']) > 0).astype(dtype=int).values

random.seed(999)
numpy.random.seed(999)
rs = 999

thresh = 0.8
thresh_val = int(X.shape[0] * thresh)
X_train, X_test = X[:thresh_val], X[thresh_val:]
Y_train, Y_test = Y[:thresh_val], Y[thresh_val:]

"""
thresh = 0.01
# disc = [any([y in x for y in ['QU', 'QD', 'CFL', 'CFY']]) for x in x_factors]
values = mutual_info_classif(X=X_train, y=Y_train, discrete_features='auto')
fs_mask = values >= thresh
"""
"""
thresh = 0.1
values = numpy.array([spearmanr(a=X_train[:, j], b=Y_train)[0] for j in range(X_train.shape[1])])
fs_mask = numpy.abs(values) >= thresh
"""
"""
thresh = 0.1
values = numpy.array([kendalltau(x=X_train[:, j], y=Y_train)[0] for j in range(X_train.shape[1])])
fs_mask = numpy.abs(values) >= thresh
"""
"""
alpha = 0.05
values = numpy.array([spearmanr(a=X_train[:, j], b=Y_train)[1] for j in range(X_train.shape[1])])
fs_mask = values <= alpha
"""
"""
alpha = 0.05
values = numpy.array([kendalltau(x=X_train[:, j], y=Y_train)[1] for j in range(X_train.shape[1])])
fs_mask = values <= alpha
"""
"""
X_train = X_train[:, fs_mask]
X_test = X_test[:, fs_mask]
"""
mkw = {'n_estimators': 1000, 'max_depth': None, 'min_samples_leaf': 1, 'random_state': rs}
# mkw = {'n_estimators': 1000, 'max_depth': 3, 'min_samples_leaf': 1, 'random_state': rs}
# mkw = {'n_estimators': 100, 'max_depth': 3, 'min_samples_leaf': 1, 'random_state': rs}
# model = RandomForestClassifier(**mkw)
model = RFECV(estimator=RandomForestClassifier(**mkw), n_jobs=-1)
model.fit(X=X_train, y=Y_train)

y_hat_train = model.predict(X=X_train)
y_hat_test = model.predict(X=X_test)

cm_train = confusion_matrix(y_true=Y_train, y_pred=y_hat_train)
cm_test = confusion_matrix(y_true=Y_test, y_pred=y_hat_test)

ac_train = accuracy_score(y_true=Y_train, y_pred=y_hat_train)
ac_test = accuracy_score(y_true=Y_test, y_pred=y_hat_test)
'''
"""
ac_train
ac_test
cm_train
cm_test
"""

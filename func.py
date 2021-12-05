#
from statsmodels.tsa.stattools import coint


#
def search_hb(data, test, test_kwargs, alpha):
    p_values = {}
    for i in range(data.shape[1]):
        for j in range(data.shape[1] - i - 1):
            _, p_value = test(ts1=data[:, i], ts2=data[:, i+j+1], **test_kwargs)
            p_values[p_value] = (i, i+j+1)
    s_keys = sorted(p_values.keys())
    keep, k, m = True, 0, len(s_keys)
    resulted = {}
    while keep and k < m:
        if s_keys[k] < (alpha / (m - k)):
            resulted[s_keys[k]] = p_values[s_keys[k]]
            k += 1
        else:
            keep = False
    return resulted


def test_eg(ts1, ts2, trend, max_lag, auto_lag):
    support, p_value, _ = coint(y0=ts1, y1=ts2, trend=trend, maxlag=max_lag, autolag=auto_lag)
    return support, p_value


def test_jo(ts1, ts2, trend, max_lag, auto_lag):
    ...


def test_po(ts1, ts2, trend, max_lag, auto_lag):
    ...

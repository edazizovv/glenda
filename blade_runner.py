#
import pandas

#
from func import search_hb, test_eg

#
d = './'
data = pandas.read_excel(d)
data = data.set_index('').sort_index()

data = data.pct_change(fill_method=None)

alpha = 0.05
kwg = {'trend': 'c', 'maxlag': None, 'autolag': 'aic'}
resulted = search_hb(data=data.values, test=test_eg, test_kwargs=kwg, alpha=alpha)

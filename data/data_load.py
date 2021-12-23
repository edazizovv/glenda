#
import os
import pandas


def load_coal_daily():

    coal_company_data = pandas.read_excel('./data/raw/coalcompany_daily.xlsx')
    coal_company_data['Date'] = pandas.to_datetime(coal_company_data['Date'])
    coal_company_data = coal_company_data.set_index('Date')
    coal_company_data = coal_company_data.rename(columns={x: '{0}__COALCOMPANY'.format(x) for x in coal_company_data.columns.values})

    coal_data = pandas.read_excel('./data/raw/coal_daily.xlsx')
    coal_data['Date'] = pandas.to_datetime(coal_data['Date'])
    coal_data = coal_data.set_index('Date')
    coal_data = coal_data.rename(columns={x: '{0}__COAL'.format(x) for x in coal_data.columns.values})

    joint_coal = coal_company_data.merge(right=coal_data, left_index=True, right_index=True, how='inner')
    joint_coal = joint_coal.sort_index()

    return joint_coal


def load_gas():

    gas_company_data = pandas.read_excel('./data/raw/gas_company.xlsx')
    gas_company_data['Date'] = pandas.to_datetime(gas_company_data['Date'])
    gas_company_data = gas_company_data.set_index('Date')
    gas_company_data = gas_company_data.rename(columns={x: '{0}__GASNCOMPANY'.format(x) for x in gas_company_data.columns.values})

    gas_data = pandas.read_excel('./data/raw/gold.xlsx')
    gas_data['Date'] = pandas.to_datetime(gas_data['Date'])
    gas_data = gas_data.set_index('Date')
    gas_data = gas_data.rename(columns={x: '{0}__GAS'.format(x) for x in gas_data.columns.values})

    joint_gas = gas_company_data.merge(right=gas_data, left_index=True, right_index=True, how='inner')
    joint_gas = joint_gas.sort_index()

    return joint_gas


def load_gold():

    golden_company_data = pandas.read_excel('./data/raw/golden_company.xlsx')
    golden_company_data['Date'] = pandas.to_datetime(golden_company_data['Date'])
    golden_company_data = golden_company_data.set_index('Date')
    golden_company_data = golden_company_data.rename(columns={x: '{0}__GOLDENCOMPANY'.format(x) for x in golden_company_data.columns.values})

    gold_data = pandas.read_excel('./data/raw/gold.xlsx')
    gold_data['Date'] = pandas.to_datetime(gold_data['Date'])
    gold_data = gold_data.set_index('Date')
    gold_data = gold_data.rename(columns={x: '{0}__GOLD'.format(x) for x in gold_data.columns.values})

    joint_gold = golden_company_data.merge(right=gold_data, left_index=True, right_index=True, how='inner')
    joint_gold = joint_gold.sort_index()

    return joint_gold


def load_silver_polymetal():

    polymetal_data = pandas.read_excel('./data/raw/polymetal.xlsx')
    polymetal_data['Date'] = pandas.to_datetime(polymetal_data['Date'])
    polymetal_data = polymetal_data.set_index('Date')
    polymetal_data = polymetal_data.rename(columns={x: '{0}__POLYMETAL'.format(x) for x in polymetal_data.columns.values})

    silver_data = pandas.read_excel('./data/raw/silver.xlsx')
    silver_data['Date'] = pandas.to_datetime(silver_data['Date'])
    silver_data = silver_data.set_index('Date')
    silver_data = silver_data.rename(columns={x: '{0}__SILVER'.format(x) for x in silver_data.columns.values})

    joint_silver = polymetal_data.merge(right=silver_data, left_index=True, right_index=True, how='inner')
    joint_silver = joint_silver.sort_index()

    return joint_silver


def load_silver_penoles():

    penoles_data = pandas.read_excel('./data/raw/penoles.xlsx')
    penoles_data['Date'] = pandas.to_datetime(penoles_data['Date'])
    penoles_data = penoles_data.set_index('Date')
    penoles_data = penoles_data.rename(columns={x: '{0}__PENOLES'.format(x) for x in penoles_data.columns.values})

    silver_data = pandas.read_excel('./data/raw/silver.xlsx')
    silver_data['Date'] = pandas.to_datetime(silver_data['Date'])
    silver_data = silver_data.set_index('Date')
    silver_data = silver_data.rename(columns={x: '{0}__SILVER'.format(x) for x in silver_data.columns.values})

    joint_silver = penoles_data.merge(right=silver_data, left_index=True, right_index=True, how='inner')
    joint_silver = joint_silver.sort_index()

    return joint_silver


def load_coal_monthly():

    bhp_data = pandas.read_excel('./data/raw/bhp.xlsx')
    bhp_data['date'] = pandas.to_datetime(bhp_data['date'])
    bhp_data['y'] = bhp_data['date'].dt.year
    bhp_data['m'] = bhp_data['date'].dt.month
    bhp_agg = bhp_data.groupby(by=['y', 'm'])['date'].min()
    bhp_data = bhp_data.set_index('date').sort_index()
    bhp_ix = [x in bhp_agg.values for x in bhp_data.index]
    bhp_data = bhp_data.loc[bhp_ix, :]
    bhp_data = bhp_data.drop(columns=['y', 'm'])
    bhp_data = bhp_data.rename(columns={x: '{0}__BHP'.format(x) for x in bhp_data.columns.values})
    bhp_data.index = bhp_data.index + pandas.offsets.MonthBegin(1) - pandas.offsets.MonthBegin(1)

    coal_data = pandas.read_excel('./data/raw/coal.xlsx')
    coal_data['date'] = pandas.to_datetime(coal_data['date'])
    coal_data = coal_data.set_index('date')
    coal_data = coal_data.rename(columns={x: '{0}__COAL'.format(x) for x in coal_data.columns.values})
    coal_data.index = coal_data.index + pandas.offsets.MonthBegin(1) - pandas.offsets.MonthBegin(1)

    joint_coal = bhp_data.merge(right=coal_data, left_index=True, right_index=True, how='inner')
    joint_coal = joint_coal.sort_index()

    return joint_coal


def load_platinum():

    angpy_data = pandas.read_excel('./data/raw/angpy.xlsx')
    angpy_data['Date'] = pandas.to_datetime(angpy_data['Date'])
    angpy_data = angpy_data.set_index('Date')
    angpy_data = angpy_data.rename(columns={x: '{0}__ANGPY'.format(x) for x in angpy_data.columns.values})

    platinum_data = pandas.read_excel('./data/raw/platinum.xlsx')
    platinum_data['date'] = pandas.to_datetime(platinum_data['date'])
    platinum_data = platinum_data.set_index('date')
    platinum_data = platinum_data.rename(columns={x: '{0}__PLATINUM'.format(x) for x in platinum_data.columns.values})

    joint_platinum = angpy_data.merge(right=platinum_data, left_index=True, right_index=True, how='inner')
    joint_platinum = joint_platinum.sort_index()

    return joint_platinum

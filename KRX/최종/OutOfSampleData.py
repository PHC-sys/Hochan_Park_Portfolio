import os
import pandas as pd
import FinanceDataReader as fdr
from collections import deque
from AnalyticUtility import rank_function
from AnalyticUtility import calculate_sharpe_ratio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
look_back_scope=60

def calculate_ret_to_mdd_squared(individual_price_data, rolling_date=15):

    """

    :param df: 단일 종목 테이블을 인자로 받음 ,
    :return:  15일 rolling ret/mdd 값을 리스톨 반환

    """
    return_to_mdd_lst = []
    close_values = deque()

    for idx in range(len(individual_price_data)-1):
        close_values.append(individual_price_data[idx])

        if len(close_values) > look_back_scope:
            close_values.popleft()

            if len(close_values) == look_back_scope:
                max_val_in_box = max(close_values)
                latest_val_in_box = close_values[-1]
                mdd_in_box = (latest_val_in_box - max_val_in_box) / max_val_in_box  # maxiumum value
                ret_in_box = (close_values[-1] - close_values[0]) / close_values[0]
                return_to_mdd_lst.append(ret_in_box / -1 * mdd_in_box**2)

    return return_to_mdd_lst , calculate_ret_to_mdd_squared

def calculate_return(database, code_lst):
    """

    :param code_lst: 200 종목 코드 리스트 (signal에서 추출된 코드)
    :param date_index:  일수
    :return: 동일 가중 포트폴리오 수익률 리스트

    """

    target_table = database # 14개 컬럼

    calculation_table = pd.DataFrame(columns=column)

    for code in code_lst:

        calculation_table.loc[str(code)] = target_table.loc[code].pct_change().values

    calculation_lst = calculation_table.sum(axis=0)

    return calculation_lst.values

sample = fdr.DataReader('005930', start = '2023-2-28')

column = sample.index

database = pd.read_csv('C:/Users/John/OneDrive/바탕 화면/퀀트관련/KRX대회/최종/baseline_submission.csv',index_col=0)

stock_lst = []

for i in database.index:
    stock_lst.append(i[1:])


merged_dataframe = pd.read_csv('C:/Users/John/OneDrive/바탕 화면/퀀트관련/KRX대회/최종/OOS_data.csv',index_col=0)

lst = []

for i in merged_dataframe.index:
    lst.append(f'{i:06d}')

merged_dataframe.index = lst

column_names = ['Column' + str(i) for i in range(61, len(merged_dataframe.columns))]

signal_dataframe = pd.DataFrame(index = stock_lst,columns=column_names)

for keys in stock_lst:  # 행: 종목코드 열: n일자의 ret/mdd값

    signal_dataframe.loc[keys] , strategy_momentum_name = calculate_ret_to_mdd_squared(merged_dataframe.loc[keys])

portfolio_total_return_lst = []


long = signal_dataframe['Column103'].sort_values(ascending=False).replace(np.nan,0).head(200)

short = signal_dataframe['Column103'].sort_values(ascending=False).replace(np.nan,0).tail(200)


drop_lst = ['090080','151860','175140']

data = signal_dataframe['Column103'].sort_values(ascending=False).replace(np.nan,0)

for i in drop_lst:
    data.loc[i] = 0

data =data.sort_values(ascending=False)
save_path = 'C:/Users/John/OneDrive/바탕 화면/퀀트관련/KRX대회/최종/'

try:
    os.mkdir(save_path)
except:
    pass

data = pd.DataFrame(data)

print(data)

df = pd.read_csv('C:/Users/John/OneDrive/바탕 화면/퀀트관련/KRX대회/최종/sample_submission.csv')

lst = []
for i in data.index:
    lst.append('A'+i)

data.index = lst
print(data.index)

data['순위'] = data['Column103'].rank(method='first', ascending=False).astype('int')
data['종목코드'] = data.index
print(data)
baseline_submission = df[['종목코드']].merge(data[['종목코드', '순위']], on='종목코드', how='left')
print(baseline_submission)
baseline_submission.to_csv(save_path + 'final_submission.csv', index=False)
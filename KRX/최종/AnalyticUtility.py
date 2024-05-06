from collections import deque
import pandas as pd
import numpy as np

truncation_pt = 0.01
truncation = int(1000*truncation_pt)

#Log 남긴함수
def rank_function(df):
    """

    :param df: 특정 날의 Signal 값을 가지는 행이 종목 코드는 pandas series
    :return: 롱 값 , 숏 값을 value로 가지는 dictionary

    """

    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)

    return {'long': df.iloc[truncation:200+truncation].index, 'short': df.iloc[-200-truncation:-truncation].index} , rank_function

#Log 남긴함수
def rank_for_hedging_function(df):
    """

    :param df: 특정 날의 Signal 값을 가지는 행이 종목 코드는 pandas series
    :return: 롱 값 , 숏 값을 value로 가지는 dictionary

    """

    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)

    return {'long': df.iloc[truncation:200+truncation].index, 'short': df.iloc[200+100+truncation:400+100+truncation].index} , rank_for_hedging_function


def calculate_return(database, code_lst, date_index, rolling_date=15):
    """

    :param code_lst: 200 종목 코드 리스트 (signal에서 추출된 코드)
    :param date_index:  일수
    :return: 동일 가중 포트폴리오 수익률 리스트

    """

    target_table = database['price']
    column_name = [f'Column{i}' for i in range(rolling_date - 1)]  # 14개 컬럼

    calculation_table = pd.DataFrame(columns=column_name)

    for code in code_lst:

        calculation_table.loc[code] = target_table[code]['Close'].pct_change()[
                                      date_index + 1:date_index + rolling_date].values

    calculation_lst = calculation_table.sum(axis=0)

    return calculation_lst.values


def calculate_mdd(daily_return_lst):
    """

    :param daily_return_lst: 20일동안의 데이터로 생성한 시그널의 15일간 포트폴리오 수익률
    :return: MDD 값 반환
    """

    mdd_lst = []
    close_values = deque()

    for idx in range(len(daily_return_lst)):

        close_values.append(daily_return_lst[idx])
        max_val_in_box = max(close_values)
        latest_val_in_box = close_values[-1]
        mdd_in_box = (latest_val_in_box - max_val_in_box) / max_val_in_box
        mdd_lst.append(mdd_in_box)

    return min(mdd_lst)


def calculate_sharpe_ratio(lst,risk_free_rate=.035):

    lst2 = lst - 1

    return ((np.exp(sum(np.log(lst)))-1)*250/15 - risk_free_rate) / (np.array(lst2).std(ddof=2)*250)
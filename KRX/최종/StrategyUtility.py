from collections import deque
import numpy as np
from sklearn.linear_model import LinearRegression

look_back_scope = 60
delay_val = 1

#Log 남긴함수
def calculate_ret_to_mdd(individual_price_data, rolling_date=15):

    """

    :param df: 단일 종목 테이블을 인자로 받음 ,
    :return:  15일 rolling ret/mdd 값을 리스톨 반환

    """
    return_to_mdd_lst = []
    close_values = deque()

    for idx in range(len(individual_price_data) - rolling_date - 1):
        close_values.append(individual_price_data['price'][idx])

        if len(close_values) > look_back_scope:
            close_values.popleft()

            if len(close_values) == look_back_scope:
                max_val_in_box = max(close_values)
                latest_val_in_box = close_values[-1]
                mdd_in_box = (latest_val_in_box - max_val_in_box) / max_val_in_box  # maxiumum value
                ret_in_box = (close_values[-1] - close_values[0]) / close_values[0]
                return_to_mdd_lst.append(ret_in_box / -1 * mdd_in_box)

    return return_to_mdd_lst , calculate_ret_to_mdd

def calculate_ret_to_mdd_squared(individual_price_data, rolling_date=15):

    """

    :param df: 단일 종목 테이블을 인자로 받음 ,
    :return:  15일 rolling ret/mdd 값을 리스톨 반환

    """
    return_to_mdd_lst = []
    close_values = deque()

    for idx in range(len(individual_price_data) - rolling_date - 1):
        close_values.append(individual_price_data['price'][idx])

        if len(close_values) > look_back_scope:
            close_values.popleft()

            if len(close_values) == look_back_scope:
                max_val_in_box = max(close_values)
                latest_val_in_box = close_values[-1]
                mdd_in_box = (latest_val_in_box - max_val_in_box) / max_val_in_box  # maxiumum value
                ret_in_box = (close_values[-1] - close_values[0]) / close_values[0]
                return_to_mdd_lst.append(ret_in_box / -1 * mdd_in_box**2)

    return return_to_mdd_lst , calculate_ret_to_mdd_squared

#Log 남긴함수

def calculate_sharpe(individual_price_data, rolling_date=15):

    """

    :param df: 단일 종목 테이블을 인자로 받음 ,
    :return:  15일 rolling ret/mdd 값을 리스톨 반환

    """
    sharpe_lst = []
    close_values = deque()

    for idx in range(len(individual_price_data) - rolling_date - 1):

        close_values.append(individual_price_data['price'][idx])

        if len(close_values) > look_back_scope:
            close_values.popleft()

            if len(close_values) == look_back_scope:

                sharpe_lst.append(np.array(close_values).mean()/np.array(close_values).std(ddof=2))

    return sharpe_lst , calculate_sharpe

#Log 남긴함수
def calculate_regress_slope(individual_price_data, rolling_date=15):

    """

    :param df: 단일 종목 테이블을 인자로 받음 ,
    :return:  15일 rolling ret/mdd 값을 리스톨 반환

    """
    regression_lst = []

    close_values = deque()

    regress = LinearRegression()

    for idx in range(len(individual_price_data) - rolling_date - 1):

        close_values.append(individual_price_data['price'][idx])

        if len(close_values) > look_back_scope:

            close_values.popleft()

            if len(close_values) == look_back_scope:

                try:
                    regress.fit(np.arange(1,len(close_values)+1).reshape(-1, 1), close_values)

                except:
                    pass

                regression_lst.append(regress.coef_[0])

    return regression_lst , calculate_regress_slope




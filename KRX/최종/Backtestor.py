#Library
from StrategyUtility import *
from AnalyticUtility import *
from scipy.stats import binom_test
import pandas as pd, FinanceDataReader as fdr
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


list_file_dir = '/Users/donghanko/Desktop/StockList/'
log_file_dir = '/Users/donghanko/Desktop/LogFile/'

try:

    os.mkdir(list_file_dir)
    os.mkdir(log_file_dir)

except:

    pass

threshold = 0.5

plt.rcParams['figure.figsize'] = 17,9
sns.set_style('whitegrid')

#DataSet Upload
with open('/Users/donghanko/Desktop/total_dataset.pickle', 'rb') as f:
    database = pickle.load(f)

#Variables
rolling_date = 15
risk_free_rate = 1.035

#Market Index
kospi_index = fdr.DataReader('^KS11',start='2021-6-1',end= '2023-5-30')['Close']
kosdaq_index = fdr.DataReader('^KQ11',start='2021-6-1',end='2023-5-30')['Close']

#Global Method
def min_max_scaler(lst):

    return (lst-min(lst))/max(lst)


def sorting_type_describer(x):
    if x=='rank_function':
        return 'Rank'

    if x=='rank_for_hedging_function':
        return 'Hedging'


class Backtester:
    def __init__(self,database):

        self.database = database     #database

        self.price_data_set = self.database['adj_close']  #price data

        self.StockList = list(self.price_data_set.keys())  #stock list

        self.finance_data_set = self.database['KOR_fs']   #finance data

        self.carry_data_set = self.database['carry']   #carry data

        self.liq_liab_data = pd.read_csv('/Users/donghanko/Desktop/liq_liab.csv',index_col=0)


    def strategy_space(self):

        """

        :return: 전략 호출 & table: 행 (종목코드) 열 (날짜) -> 시그널 값

        """
        #scope 기간은 look_back_scope - 478 (= 494-rolling_date-1)

        data = (self.liq_liab_data)/self.liq_liab_data.iloc[0]



        column_names = ['Column' + str(i) for i in range(look_back_scope+1, 479)]

        # 행이 종목 코드이고 열이 15일 이후 날짜
        merged_data = pd.DataFrame(index=self.price_data_set.keys(), columns=column_names)

        #종목당 가격 데이터에 접근 후 calculate_ret_to_mdd function에 toss
        for keys in list(self.price_data_set.keys()):  # 행: 종목코드 열: n일자의 ret/mdd값

            merged_data.loc[keys] , strategy_momentum_name = calculate_ret_to_mdd_squared(self.price_data_set[keys])

            #merged_data.loc[keys]*= data.loc[int(keys)].values[0]

        return merged_data,strategy_momentum_name


    def calculate_pnl(self):
        print('Start')

        merged_data , strategy_name = self.strategy_space()

        print(merged_data)

        date_sequence = look_back_scope + 1

        portfolio_total_return_lst = []
        portfolio_sharpe_lst = []
        portfolio_mdd_lst = []
        portfolio_vol_lst = []


        while date_sequence <= len(self.price_data_set[self.StockList[0]]) - rolling_date - 1:

            dict , sort_type = rank_function(merged_data[f'Column{date_sequence}'])
            #dict = rank_for_hedging_function(merged_data[f'Column{date_sequence}'])

            for keys in dict.keys():

                if keys == 'long':

                    long_ret_lst = calculate_return(database,dict[keys], date_sequence)

                else:

                    short_ret_lst = np.array(-1*(calculate_return(database, dict[keys], date_sequence)))


            portfolio_ret_lst = np.array([(x + y)/400 + 1  for x,y in zip(long_ret_lst,short_ret_lst)])

            portfolio_sharpe_lst.append(calculate_sharpe_ratio(portfolio_ret_lst))

            portfolio_mdd_lst.append(calculate_mdd(portfolio_ret_lst))

            portfolio_total_return_lst.append((np.exp(sum(np.log(portfolio_ret_lst)))-1)*250/15 -.035)

            portfolio_vol_lst.append(np.array(portfolio_ret_lst).std(ddof=2))

            print(f'Done{date_sequence-look_back_scope}')

            date_sequence += 1

        print(f'Final Analytics: Mean {np.array(portfolio_sharpe_lst).mean()} , Std {np.array(portfolio_sharpe_lst).std()}')

        domain =  pd.to_datetime(self.price_data_set[self.StockList[0]].index[look_back_scope+1:-rolling_date])

        success_count = sum(1 for value in portfolio_sharpe_lst if value >= threshold)

        print(success_count,len(portfolio_sharpe_lst))

        p_value = binom_test(success_count, len(portfolio_sharpe_lst), p=threshold, alternative='less')

        alpha = 0.05

        # p-value를 유의수준과 비교하여 검정 결과를 출력
        if p_value < alpha:
            print(f"영가설(H0)을 기각합니다. 주어진 데이터는 {threshold} 이상일 확률이 유의미하게 낮습니다.")
        else:
            print(f"영가설(H0)을 채택합니다. 주어진 데이터는 {threshold} 이상일 확률이 유의미하게 높습니다.")

        fig = plt.figure()
        ax = fig.add_subplot(4,1,1)
        ax1 = fig.add_subplot(4,1,2)
        ax2 = fig.add_subplot(4,1,3)
        ax3 = fig.add_subplot(4,1,4)

        ax.plot(domain, portfolio_total_return_lst)
        ax.set_title('Return')

        ax1.plot(min_max_scaler(kospi_index),label='KOSPI Index')
        ax1.plot(min_max_scaler(kosdaq_index),label='KOSDAQ Index')
        ax1.set_title('Index')
        ax1.legend()

        ax2.plot(domain,portfolio_sharpe_lst)
        ax2.axhline(y=np.array(portfolio_sharpe_lst).mean(), color='r' ,lw=.5,label= f'Mean{np.array(portfolio_sharpe_lst).mean()}')
        ax2.axhline(y=threshold,color='b',lw=.5,label='Threshold')
        ax2.set_title('Sharpe Ratio')
        ax2.legend()

        ax3.plot(domain,portfolio_mdd_lst)
        ax3.set_title('Maximum Draw Down')

        plt.tight_layout()
        plt.show()

        print(f'Correlation: {np.corrcoef(kospi_index[-len(portfolio_sharpe_lst):], portfolio_sharpe_lst)[0][1]}')

        #log 기록 남기기
        log_record = {'Strategy': strategy_name.__name__  ,'LookBackScope':look_back_scope,'Truncation':truncation_pt,'SortingType':
                      sorting_type_describer(sort_type.__name__), 'Peformance':pd.DataFrame(portfolio_sharpe_lst).describe(),
                      'CorrelationWithMarket':np.corrcoef(kospi_index[-len(portfolio_sharpe_lst):], portfolio_sharpe_lst)[0][1]}

        print('Log Record: ', log_record)

        file_path = log_file_dir + f'log_record { strategy_name.__name__ }.pickle'

        #로그 기록 로컬에 저장

        with open(file_path,'wb') as f:
            pickle.dump(log_record,f)

        merged_data.to_csv(list_file_dir + f'our_data {max(portfolio_sharpe_lst):03f}.csv')
        print('Data Saved.')


if __name__=='__main__':
    start_time = time.time()

    test = Backtester(database)
    test.calculate_pnl()

    end_time = time.time()
    total_time = end_time-start_time
    print(f'{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{total_time%60}')


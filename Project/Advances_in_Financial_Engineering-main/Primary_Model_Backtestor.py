#Import and Settings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from collections import deque
from itertools import accumulate
import bisect
import warnings
plt.rcParams['figure.figsize'] = 17,9
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

#Data Loads
kq150_data = pd.read_csv(r'C:\Users\John\OneDrive\바탕 화면\퀀트관련\퀀트스터디\afml Project\Advances_in_Financial_Engineering-main\KQ150.csv',index_col=0)
kq150_data.index = pd.to_datetime(kq150_data.index)
test_data = kq150_data.iloc[:200000,:]

#Utils

def parkinson_volatility(df, n):
    data_scope = deque()
    parkison_vol_lst = []
    for i in range(len(df)):
        data_scope.append(df.iloc[i, :])
        if len(data_scope) > n:
            data_scope.popleft()

        parkinson_vol = np.sqrt(
            sum(np.log(pd.DataFrame(data_scope)['High'].values / pd.DataFrame(data_scope)['Low'].values) ** 2) / (
                        4 * np.log(2) * n))
        parkison_vol_lst.append(parkinson_vol)
    return parkison_vol_lst

def local_min(df):
    local_min_lst = []
    for i in range(1,len(df)-1):
        if (df[i]<=df[i-1]) and (df[i]<df[i+1]):  # 두번째는 등호 포함 x -> 첫 local min을 찍는 point index 반환
            local_min_lst.append(i)
    return local_min_lst

def local_max(df):
    local_max_lst = []
    for i in range(1,len(df)-1):
        if (df[i] >= df[i - 1]) and (df[i] > df[i + 1]):  # 두번째는 등호 포함 x -> 첫 local max을 찍는 point index 반환
            local_max_lst.append(i)
    return local_max_lst

def functional_value(df,lst):
    """
    :param df:
    :param lst:
    :return: 함수 값을 리스트로 반환
    """
    func_val = []
    for i in lst:
        func_val.append(df[i])
    return func_val

def SMA(data,n):
    sma = deque()
    sma_lst = []
    for i in data:
        sma.append(i)
        if len(sma)>n:
            sma.popleft()
        sma_lst.append(sum(sma)/len(sma))
    return sma_lst

def find_closest(lst,val):
    """
    :param lst: 탐색 리스트
    :param val: 특정 값
    :return : 특정 값과 가장 가까운 크고 작은 값 반환
    """
    i = bisect.bisect_left(lst,val)
    if i==len(lst):
        return lst[-1],lst[-2]
    elif i==0:
        return lst[0],lst[1]
    else:
        before = lst[i-1]
        after = lst[i]
        return before,after

def count_reversion_element(df,index,thres_down,thres_up):
    """
    :param df: 함수
    :param index: index (전략에서는 local min을 가지는 index를 취할 것)
    :param thres: threshold 기반으로 시그널 생성
    :return: threshold를 뚫는 index 반환
    """
    left_index = None
    right_index = None
    for i in range(index-1,-1,-1):
        if (df[i]<thres_down) and (df[i-1]>thres_down):
            left_index = i-1
            break

    for i in range(index, len(df)-1):
        if (df[i]<thres_up) and (df[i+1]>thres_up):
            right_index = i+1
            break
    return left_index,right_index

def StochasticOscillator(data, n=5, m=3, t=3):
    """
    Generating Stoc Indicator
    :param data:
    :param n:
    :param m:
    :param t:
    :return: slow_k for entrance and slow_d for exit
    """
    fast_k = []
    data_lst = deque()
    for i in range(len(data)):
        data_lst.append(data.loc[data.index[i],['High','Low','Close','Open']])
        if len(data_lst)>n:
            data_lst.popleft()
        high = max(pd.DataFrame(data_lst)['High'])
        low = min(pd.DataFrame(data_lst)['Low'])
        close = pd.DataFrame(data_lst)['Close'][-1]
        open = pd.DataFrame(data_lst)['Open'][0]
        fast_k.append((close-open)/(high-low)*100)
    slow_k = SMA(fast_k,m)
    slow_d = SMA(slow_k,t)
    return fast_k,slow_k,slow_d


class BackTestor:
    def __init__(self, data, scope_length, maximum_holding_period):
        """
        :param data:
        :param n: scope length of calculating signal
        :param m: maximum bar of holding position
        """
        self.data = data
        self.scope_length = scope_length
        self.maximum_holding_period = maximum_holding_period

    def create_daily_bar_no(self):
        """
        :return: create dataframe of daily bar no
        """

        df  =  self.data.groupby([self.data.index.date]).count()
        daily_ticks  =  pd.DataFrame(index = df.index)
        daily_ticks['no_ticks'] = list(accumulate(df['Open']))
        return daily_ticks

    def in_sample_calculate_pnl(self, thres , in_out_ratio=.8, maximum_trading = 3):
        """
        :param thres: Threshold for seeking as over sold
        :param in_out_ratio: in-sample and out-sample ratio
        :return:
        """
        df = pd.DataFrame()
        daily_ticks = self.create_daily_bar_no()
        pnl_lst = []
        win_lst = []
        lose_lst = []
        daily_lst = []
        mdd_lst = []
        entrance = 0
        iter = 0
        pnl = 1
        for k in range(490,int(len(daily_ticks)*in_out_ratio)):
            try:
                if k==0:
                    test_sample = self.data[:daily_ticks.iloc[k,0]]
                else:
                    test_sample = self.data[daily_ticks.iloc[k,0]:daily_ticks.iloc[k+1,0]]
            except:
                pass
            parkinson_vol = parkinson_volatility(test_sample,self.scope_length)
            fast_k, a, b = StochasticOscillator(test_sample,self.scope_length,1,1)
            #fast_k = SMA(fast_k,3)
            target_lst = [x for x in local_min(fast_k) if fast_k[x]<= -thres]
            #print(len(target_lst))

            entrance_lst =[]
            remove_mutual = []
            daily_returns = 1
            #target_lst = [x for x in target_lst if (x>60) and (x<len(test_sample)-60)]  #Time Adjustment
            #print(target_lst)
            targeting_lst = []
            for i in range(len(target_lst)-1):
                if max(fast_k[target_lst[i] : target_lst[i+1]])<=-10:
                    targeting_lst.append(target_lst[i+1])
                if target_lst[i] in targeting_lst and (max(fast_k[target_lst[i] : target_lst[i+1]])<=-10):
                    targeting_lst.remove(target_lst[i])


            for idx in targeting_lst:
                left , right = count_reversion_element(fast_k,idx,0,-55)
                if (left is None) or (right is None):
                    continue

                if  (abs(idx-right)<4) and (abs(idx-left) > abs(idx-right))  and (right not in remove_mutual) and (parkinson_vol[idx]>.0006): #Trigger conditioning

                    remove_mutual.append(right)
                    entrance_lst.append(right)

                    if len(entrance_lst)> maximum_trading:
                        break

                    try:
                        entrance_price = test_sample['Open'][right + 1]
                        for i in range(1, self.maximum_holding_period-1):
                            print("for loop: ", i)
                            numerator = (test_sample['Close'][right+1+i] - entrance_price -.2)
                            return_on_signal = (numerator/entrance_price)

                            if (return_on_signal>=.015) or (return_on_signal<=-.0015):
                                numerator = numerator- .2   #손절 라인을 넘으면 그 다음 시가에서 매도
                                return_on_signal =  numerator/entrance_price
                                print('Return from the Signal is: ', return_on_signal) #여기 추가
                                df.loc[iter, 'Entrance'] = test_sample.index[right + 1]
                                df.loc[iter, 'Clearing'] = test_sample.index[right + i + 1]
                                print(test_sample.index[idx], test_sample.index[right + 1],
                                      test_sample.index[right + i + 1])
                                break

                            if i == self.maximum_holding_period-2:
                                return_on_signal =  (numerator-.2)/entrance_price

########################
                            print('Return from the Signal is: ', return_on_signal)
                            df.loc[iter, 'Entrance'] = test_sample.index[right + 1]
                            df.loc[iter, 'Clearing'] = test_sample.index[right + i + 1]
                            print(test_sample.index[idx], test_sample.index[right + 1], test_sample.index[right + i + 1])
                        
                        print("last: ", i)
                        iter+=1
                        daily_returns += (return_on_signal)
                        pnl += (return_on_signal)
                        
########################
                    except:   #만약 시그널 이후 maximum holding이 장 마감 청산 기간을 넘길 경우
                        try:
                            entrance_price = test_sample['Open'][right + 1]
                            iter+=1
                            for i in range(self.maximum_holding_period - 1):

                                numerator = (test_sample['Close'][right + 1 + i] - entrance_price - .2)
                                return_on_signal =  (numerator / entrance_price)

                                if (return_on_signal >= .015) or (return_on_signal <= -.0015):
                                    numerator = numerator - entrance_price - .2
                                    return_on_signal =  numerator / entrance_price
                                    print('Return from the Signal is: ', return_on_signal) #여기 추가
                                    df.loc[iter, 'Entrance'] = test_sample.index[right + 1]
                                    df.loc[iter, 'Clearing'] = test_sample.index[right + i + 1]
                                    print(test_sample.index[idx], test_sample.index[right + 1],
                                          test_sample.index[right + i + 1])
                                    break

                                if i == len(test_sample) - 2:
                                    return_on_signal =  (numerator - .2) / entrance_price

                                print('Return from the Signal is: ', return_on_signal)
                                df.loc[iter,'Entrance'] = test_sample.index[right + 1]
                                df.loc[iter,'Clearing'] = test_sample.index[right + i + 1]
                                print(test_sample.index[idx],test_sample.index[right + 1],test_sample.index[right + i + 1])
                                
                            daily_returns += (return_on_signal)
                            pnl += (return_on_signal)
                        except:
                            print('Signal Error has been found')

            entrance+=1
            if daily_returns > 1:    #Append winning date
                win_lst.append(1)
            if daily_returns < 1:
                lose_lst.append(1)
            print(daily_returns)
            daily_lst.append(daily_returns-1)
            print('Cumlative PNL: ',pnl)

            pnl_lst.append(pnl)
            peak = max(pnl_lst)
            mdd_lst.append(1-peak/pnl)

            print(f'Process {k+1} Done')

        print('MDD: ', -min(mdd_lst)*100)
        print('Sharpe: ',np.array(daily_lst).mean()/np.array(daily_lst).std()*np.sqrt(len(daily_lst)))
        print('Total Entrance: ',entrance)
        print('Win Ratio: ', len(win_lst)/(len(win_lst)+len(lose_lst)))
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        ax1 = fig.add_subplot(2,1,2)
        plt.title('PNL Graph')
        ax.plot(pnl_lst)
        ax1.plot(mdd_lst)
        plt.show()
        df.to_csv(r'C:\Users\John\OneDrive\바탕 화면\퀀트관련\퀀트스터디\afml Project\Advances_in_Financial_Engineering-main\entrance_exit_in.csv')


backtestor = BackTestor(test_data,scope_length=10 , maximum_holding_period = 5)
backtestor.in_sample_calculate_pnl(thres=70)











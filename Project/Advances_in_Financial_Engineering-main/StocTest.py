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
#kp200_data = pd.read_csv(r"C:\Users\Qraft\Desktop\KP200.csv",index_col=0)
#kp200_data.index = pd.to_datetime(kp200_data.index)

kq150_data = pd.read_csv('/Users/donghanko/Downloads/KQ150.csv',index_col=0)
kq150_data.index = pd.to_datetime(kq150_data.index)

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

        df = self.data.groupby([self.data.index.date]).count()
        daily_ticks = pd.DataFrame(index = df.index)
        daily_ticks['no_ticks'] = list(accumulate(df['Open']))
        return daily_ticks

    def in_sample_calculate_pnl(self, thres , in_out_ratio=.8, maximum_trading = 1):
        """
        :param thres: Threshold for seeking as over sold
        :param in_out_ratio: in-sample and out-sample ratio
        :return:
        """
        daily_ticks = self.create_daily_bar_no()
        pnl_lst = []
        win_lst = []
        entrance = 0
        pnl = 1
        iter = 0
        for k in range(490,int(len(daily_ticks)*in_out_ratio)):
            if k==0:
                test_sample = self.data[:daily_ticks.iloc[k,0]]
            else:
                test_sample = self.data[daily_ticks.iloc[k,0]:daily_ticks.iloc[k+1,0]]

            parkinson_vol = parkinson_volatility(test_sample,self.scope_length)
            fast_k, a, b = StochasticOscillator(test_sample,self.scope_length,1,1)
            #fast_k = SMA(fast_k,3)
            target_lst = [x for x in local_min(fast_k) if fast_k[x]<= -thres]


            entrance_lst =[]
            remove_mutual = []
            daily_returns = 1
            target_lst = [x for x in target_lst if (x>30)]  #Time Adjustment
            #print(target_lst)
            targeting_lst = []
            for i in range(len(target_lst)-1):
                if max(fast_k[target_lst[i] : target_lst[i+1]])<=-20:
                    targeting_lst.append(target_lst[i+1])
                if (target_lst[i] in targeting_lst) and (max(fast_k[target_lst[i] : target_lst[i+1]])<=-30):
                    targeting_lst.remove(target_lst[i])

            print(len(targeting_lst))

            for idx in targeting_lst:
                left , right = count_reversion_element(fast_k,idx,0,-20)
                if (left is None) or (right is None):
                    continue

                if  (abs(idx-right)<5) and (abs(idx-left) > abs(idx-right))  and (right not in remove_mutual) and (parkinson_vol[idx]>.0006): #Trigger conditioning

                    remove_mutual.append(right)
                    entrance_lst.append(right)
                    print('Entrance: ',right)
                    if len(entrance_lst)> maximum_trading:
                        break
                    try:
                        entrance_price = test_sample['Open'][right + 1]
                        for i in range(self.maximum_holding_period-1):

                            numerator = (test_sample['Close'][right+1+i] - entrance_price -.2)
                            return_on_signal = 1+(numerator/entrance_price)
                            if (return_on_signal>=1.008) or (return_on_signal<=0.996):
                                numerator = test_sample['Open'][right+2+i] - entrance_price- .2
                                return_on_signal = 1+ numerator/entrance_price
                                break

                            if i == self.maximum_holding_period-2:
                                numerator = (test_sample['Close'][right+1+i] - entrance_price -.2)
                                return_on_signal = 1 + (numerator-.2)/entrance_price


                        print('Return from the Signal is: ', return_on_signal)
                        iter+=1
                        daily_returns *= (return_on_signal)
                        pnl *= (return_on_signal)

                    except:   #만약 시그널 이후 maximum holding이 장 마감 청산 기간을 넘길 경우
                        try:
                            entrance_price = test_sample['Open'][right + 1]
                            iter+=1
                            for i in range(self.maximum_holding_period - 1):

                                numerator = (test_sample['Close'][right + 1 + i] - entrance_price - .2)
                                return_on_signal = 1 + (numerator / entrance_price)
                                if (return_on_signal >= 1.008) or (return_on_signal <= 0.996):
                                    numerator = test_sample['Open'][right + 2 + i] - entrance_price - .2
                                    return_on_signal = 1+  numerator / entrance_price
                                    break

                                if i == len(test_sample) - 2:
                                    numerator = (test_sample['Close'][len(test_sample)]-entrance_price-.2)
                                    return_on_signal = 1 + (numerator - .2) / entrance_price
                            print('Return from the Signal is: ', return_on_signal)

                            daily_returns *= (return_on_signal)
                            pnl *= (return_on_signal)
                        except:
                            print('Signal Error has been found')
            entrance+=len(entrance_lst)

            if daily_returns >1:    #Append winning date
                win_lst.append(1)


            print('Cumlative PNL: ',pnl)
            pnl_lst.append(pnl)
            print(f'Process {k+1} Done')

        print('Total Entrance: ',entrance)
        print('Win Ratio: ', len(win_lst)/iter)
        plt.title('PNL Graph')
        plt.plot(pnl_lst)
        plt.show()




backtestor = BackTestor(kq150_data,scope_length=10,maximum_holding_period = 6)
backtestor.in_sample_calculate_pnl(thres=80)












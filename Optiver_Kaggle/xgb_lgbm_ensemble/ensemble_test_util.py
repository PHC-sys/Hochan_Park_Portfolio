import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgbm
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import util


'''
Optiver Backtestor

<Backtestor Process>

1. Data에서 id 별로 뽑아내서 학습: 뽑아낼 때 loc, condtion
2. model 저장

* Ensemble : xgb*.8 + lgbm* .2

3. 학습한 모델로 wap60 채워 넣어야함(index로 Mapping)
    ex. test_data.loc[test_data.stock_id == 0, 'wap_60'] = stock_0_test['wap_60']
    
4. wap_60 > index_60 calculation
5. index data 계산 미리
6. target = (wap_60/wap - index_60/index)*10000

===================================================================

<Data 관련>

7. train_data preprocessing: date_id == 478,479,480 train_data에서 분리해야할듯

8. revealed_target : 전날 시계열 데이터 > gbm base(h0)로 이용 가능하려나 ?

9. seconds_in_bucket >= 490 : train 할때 빼버릴것 (wap_60이 없어)
'''

def generate_clustered_stock_data(train_data, cluster_results = {i:[j] for i, j in zip(range(0,200), range(0,200))}):
    '''
    Cluster 결과에 따라 train_data 나눠주는 함수
    train_data > clustered_stock_data
    '''
    clustered_stock_data={}
    for key in list(cluster_results.keys()):
        stock_id_list = cluster_results[key]
        stock_data = train_data.query(f'stock_id == {stock_id_list}')
        clustered_stock_data[tuple(stock_id_list)] = stock_data
        
    return clustered_stock_data


def preprocessing(stock_data, scale = False, target = 'wap_60'):
    '''
    전처리
    stock_data > X_train, X_test, y_train, y_test
    '''
    if target == 'wap_60':
    
        stock_data.loc[:, 'wap_60'] = stock_data.loc[:,'wap'].shift(-6)
        stock_data = stock_data.query('seconds_in_bucket <= 480')
        stock_data = stock_data.query('date_id < 478')
        stock_data = stock_data.drop(columns=['time_id','row_id','far_price','near_price']).dropna()

        X_data = stock_data.drop(columns=['target', 'wap_60']).dropna()
        y_data = stock_data.loc[:,'wap_60']
        target = stock_data['target']
        
        X_train = X_data[:-2000]
        X_test = X_data[-2000:]
        y_train = y_data[:-2000]
        y_test = y_data[-2000:]
        target_test = target[-2000:]
        
        if scale == True:
            scaler = MinMaxScaler()
            X_train[:] = scaler.fit_transform(X_train)
            X_test[:] = scaler.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=X_data.columns)
            X_test = pd.DataFrame(X_test, columns=X_data.columns)
        
        return X_train, X_test, y_train, y_test, target_test
    
    elif target == 'target':
    
        stock_data = stock_data.query('seconds_in_bucket <= 480')
        stock_data = stock_data.query('date_id < 478')
        stock_data = stock_data.drop(columns=['time_id','row_id','far_price','near_price']).dropna()

        X_data = stock_data.drop(columns=['target']).dropna()
        y_data = stock_data.loc[:,'target']
        target = stock_data['target']
        
        X_train = X_data[:-2000]
        X_test = X_data[-2000:]
        y_train = y_data[:-2000]
        y_test = y_data[-2000:]
        target_test = target[-2000:]
        
        if scale == True:
            scaler = MinMaxScaler()
            X_train[:] = scaler.fit_transform(X_train)
            X_test[:] = scaler.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=X_data.columns)
            X_test = pd.DataFrame(X_test, columns=X_data.columns)
        
        return X_train, X_test, y_train, y_test, target_test


def generate_model(stock_data, scale = False, model_name = 'xgb', target='wap_60'): #params
    '''
    각 stock_id 마다 (xgb) model 생성
    stock_data > wap_60
    '''
    #models = {}
    #mae_dict = {}
    #stock_id_list = list(stock_data.stock_id.value_counts().sort_index().index)
    
    if model_name == 'xgb':
        xgbr = xgb.XGBRegressor(objective ='reg:absoluteerror', n_estimators = 100, seed = 123)
        X_train, X_test, y_train, y_test, target_test = preprocessing(stock_data, scale=scale, target= target)
        #print(X_test)
        model = xgbr.fit(X_train, y_train)
        pred = model.predict(X_test) #wap_60
        
        pred_target = util.calculate_target(X_test, pred, target)
        
        mae = abs(pred_target-target_test).mean()
    
    elif model_name == 'lgbm':
        lgbmr = lgbm.LGBMRegressor(objective ='regression',metric='mean_absolute_error', n_estimators = 100, seed = 123)
        X_train, X_test, y_train, y_test, target_test = preprocessing(stock_data)
        #print(X_test)
        model = lgbmr.fit(X_train, y_train)
        pred = model.predict(X_test) #wap_60/target
        
        pred_target = util.calculate_target(X_test, pred, target)
        
        mae = abs(pred_target-target_test).mean()
    
    #models[stock_id_list] = model
    #mae_dict[stock_id_list] = mae
    
    return model, mae


def generate_target(test_data, model, target = 'wap_60'):
    '''
    test_data에 wap_60 추가
    '''
    #clustered_test_data = generate_clustered_stock_data(test_data)
    pred = model.predict(test_data)
    #print(pred)
    
    pred_target = util.calculate_target(test_data, pred, target)

    return pred_target

#load
with open('./data/train.pickle', 'rb') as f:
    train_data = pickle.load(f)

test_data = pd.read_csv('./example_test_files/test.csv')

if __name__ == "__main__":
    #cluster_results = {0:[j for j in range(0,200)]}
    clustered_stock_data = generate_clustered_stock_data(train_data)
    clustered_test_data = generate_clustered_stock_data(test_data)
    models = {}
    mae_dict = {}
    submission = pd.DataFrame()
    #params = {'objective' :'reg:squarederror','n_estimators' : 30,'seed' : 123}
    counter = 0
    for key in list(clustered_stock_data.keys()):
        print(counter)
        stock_data = clustered_stock_data[key]
        model, mae = generate_model(stock_data)
        models[key] = model
        mae_dict[key] = mae
        
        test_data = clustered_test_data[key].drop(columns=['time_id','row_id','far_price','near_price']).dropna()
        #print('test_data:', test_data)
        target = generate_target(test_data, model)
        target = pd.to_numeric(target)
        #print(target.dtype)
        submission = pd.concat([submission,target])
        #print(submission)
        counter += 1
        '''
        if counter == 10:
            break
        '''
        
    submission = submission.sort_index()
    real_value = train_data.query('date_id>=478')['target'].values
    print(real_value)
    print(submission)
    print(abs(submission.values - real_value).mean())
    submission.to_csv('sample_submssion_xgb.csv')
        
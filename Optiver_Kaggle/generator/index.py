import pandas as pd
import pickle


def wap_mean(train_data):
    index_list = []
    for i in range(0,481):
        print('date:',i)
        for j in range(0, 541, 10):
            index = train_data.query(f'date_id == {i} and seconds_in_bucket == {j}')['wap'].mean()
            index_list.append(index)
    
    return index_list
        

def wap_matched_size(train_data):
    index_list = []
    for i in range(0,481):
        print('date:',i)
        for j in range(0, 541, 10):
            data = train_data.query(f'date_id == {i} and seconds_in_bucket == {j}')
            index = (data['wap']*data['matched_size']).sum()/(data['matched_size']).sum()
            index_list.append(index)
    
    return index_list

def real_index(train_data, raw_weights):
    index_list = []
    for i in range(0,481):
        print('date:',i)
        for j in range(0, 541, 10):
            weights = []
            data = train_data.query(f'date_id == {i} and seconds_in_bucket == {j}')
            for id in range(200):
                if id in data.stock_id.values:
                    weights.append(raw_weights[id])

            index = np.array(data['wap'].values*weights).sum()/np.array(weights).sum()
            index_list.append(index)
    
    return index_list

#load
with open('./data/train.pickle', 'rb') as f:
    train_data = pickle.load(f)

#load
with open('./data/index_weights.pickle', 'rb') as f:
    raw_weights = pickle.load(f)

index_list = real_index(train_data, raw_weights)
print(index_list)
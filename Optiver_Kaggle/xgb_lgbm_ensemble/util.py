def index_expansion(index):
    l = []
    for i in index:
        for j in range(0,200):
            l.append(i)
    
    return i

def calculate_target(data, pred, target):
    
    if target == 'wap_60':
        #print(data)
        data_copy = data.copy()
        data_copy['wap_60'] = pred
        wap = data_copy.wap
        index = data_copy.groupby(['date_id','seconds_in_bucket'])['wap'].mean().values
        index_60 = data_copy.groupby(['date_id','seconds_in_bucket'])['wap_60'].mean().values
        #print('index:',index)
        
        index = index_expansion(index)
        index_60 = index_expansion(index_60)

        data_copy.loc[:,'index'] = index
        data_copy.loc[:,'index_60'] = index_60
        
        pred_target = (data_copy.wap_60/wap - data_copy.index_60/data_copy['index'])*10000
        #print(data_copy.wap_60, wap, data_copy.index_60, data_copy['index'], pred_target)
    
    elif target == 'target':
        #print(data)
        data_copy = data.copy()
        data_copy['target'] = pred
        
        pred_target = data_copy.target

    return pred_target
import numpy as np
import pandas as pd
import pickle

# load
with open('./data/stock_0.pickle', 'rb') as f:
    stock_0 = pickle.load(f)
    
#load
with open('./data/train.pickle', 'rb') as f:
    train_data = pickle.load(f)

state_matrix_0 = pd.DataFrame()
state_matrix_0['target'] = stock_0.target
state_matrix_0 = state_matrix_0.reset_index(drop=True)

def genearte_p_matrix(state_matrix, q):
    state_list = np.array([])
    #state_list = np.append(state_list, 1)
    for i in range(len(state_matrix)):
        target = state_matrix.iloc[i].values[0]
        #print(target, type(target))
        if target <= q[0] :
            state = 5
        elif target >= q[4] :
            state = 4
        elif target > q[0] and target <= q[1] :
            state = 3
        elif target > q[1] and target <= q[2] :
            state = 1
        elif target > q[2] and target <= q[3] :
            state = 0
        elif target > q[3] and target < q[4] :
            state = 2
        state_list = np.append(state_list, state)

    state_matrix['state_i'] = state_list
    state_matrix['state_j'] = state_matrix.state_i.shift(-6)
    state_matrix.drop(columns=['target'], inplace=True)
    state_matrix.dropna(inplace=True)

    p_matrix_0 = []
    state = [0,1,2,3,4,5]

    for i in state:
        #print(i)
        p_row = []
        denominator = state_matrix.state_i.value_counts().sort_index().values[i] # the number of state i
        sum_num = 0
        for j in state:
            numerator = len(state_matrix.query(f'state_i == {i} and state_j =={j}'))
            sum_num += numerator
            conditional_prob = numerator/denominator
            p_row.append(round(conditional_prob,2))
        
        #print(denominator, sum_num)
        #print(sum(np.array(p_row)))
        
        p_matrix_0.append(p_row)
    
    p_matrix_0 = np.array(p_matrix_0)
    
    return p_matrix_0, q

'''
p_matrix = genearte_p_matrix(state_matrix_0, (-20, -5, 0, 5, 20))
print('For', p_matrix[1],':\n', p_matrix[0]*100)

state_matrix_0 = pd.DataFrame()
state_matrix_0['target'] = stock_0.target
state_matrix_0 = state_matrix_0.reset_index(drop=True)

q1 = float(stock_0.target.quantile(q=0.01, interpolation='nearest'))
q2 = float(stock_0.target.quantile(q=0.05, interpolation='nearest'))
q3 = float(stock_0.target.quantile(q=0.5, interpolation='nearest'))
q4 = float(stock_0.target.quantile(q=0.95, interpolation='nearest'))
q5 = float(stock_0.target.quantile(q=0.99, interpolation='nearest'))
q = (q1,q2,q3,q4,q5)

p_matrix = genearte_p_matrix(state_matrix_0, q)
print('For', p_matrix[1],':\n', p_matrix[0]*100)

stock_1 = train_data[train_data['stock_id'] == 1]
    
state_matrix_1 = pd.DataFrame()
state_matrix_1['target'] = stock_1.target
state_matrix_1 = state_matrix_1.reset_index(drop=True)

p_matrix = genearte_p_matrix(state_matrix_1, (-40, -20, 0, 20, 40))
print('For', p_matrix[1],':\n', p_matrix[0]*100)


state_matrix_1 = pd.DataFrame()
state_matrix_1['target'] = stock_1.target
state_matrix_1 = state_matrix_1.reset_index(drop=True)

q1 = float(stock_1.target.quantile(q=0.01, interpolation='nearest'))
q2 = float(stock_1.target.quantile(q=0.05, interpolation='nearest'))
q3 = float(stock_1.target.quantile(q=0.5, interpolation='nearest'))
q4 = float(stock_1.target.quantile(q=0.95, interpolation='nearest'))
q5 = float(stock_1.target.quantile(q=0.99, interpolation='nearest'))
q = (q1,q2,q3,q4,q5)

p_matrix = genearte_p_matrix(state_matrix_1, q)
print('For', p_matrix[1],':\n', p_matrix[0]*100)
'''

state_matrix_0 = pd.DataFrame()
state_matrix_0['target'] = train_data.target
state_matrix_0 = state_matrix_0.reset_index(drop=True)

p_matrix = genearte_p_matrix(state_matrix_0, (-20, -5, 0, 5, 20))
print('For', p_matrix[1],':\n', p_matrix[0]*100)
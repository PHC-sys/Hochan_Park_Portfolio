import ensemble_test_util as ens_util
import pickle
import pandas as pd
import matplotlib.pyplot as plt

#load
with open('./data/train.pickle', 'rb') as f:
    train_data = pickle.load(f)

test_data = pd.read_csv('./example_test_files/test.csv')

if __name__ == "__main__":
    cluster_results = {0:[j for j in range(0,200)]}
    clustered_stock_data = ens_util.generate_clustered_stock_data(train_data, cluster_results)
    clustered_test_data = ens_util.generate_clustered_stock_data(test_data, cluster_results)
    models = {}
    mae_dict = {}
    submission = pd.DataFrame()
    #params = {'objective' :'reg:squarederror','n_estimators' : 30,'seed' : 123}
    counter = 0
    for key in list(clustered_stock_data.keys()):
        print(counter)
        stock_data = clustered_stock_data[key]
        model1, mae1 = ens_util.generate_model(stock_data, scale=False, model_name='xgb',target='target' )
        model2, mae2 = ens_util.generate_model(stock_data, scale=False, model_name='lgbm',target='target')
        models[key] = (model1, model2)
        mae_dict[key] = (mae1, mae2)
        
        test_data = clustered_test_data[key].drop(columns=['time_id','row_id','far_price','near_price']).dropna()
        #print('test_data:', test_data)
        target1 = ens_util.generate_target(test_data, model1, target='target')
        target2 = ens_util.generate_target(test_data, model2, target='target')
        #print(target1)
        target = pd.to_numeric(target1*0.5 + target2*0.5)
        #print(target)
        submission = pd.concat([submission,target])
        #print(submission)
        counter += 1
        '''
        if counter == 1:
            break
        '''
    
    submission = submission.sort_index()
    real_value = train_data.query('date_id>=478')['target'].values
    #print(real_value)
    #print(submission)
    
    for mae in mae_dict.values():
        if mae[0] < mae[1]:
            print("Val_MAE:", mae)
    
    #print(mae_dict.values().mean())
    test_error = abs(submission.values - real_value)
    print(test_error.mean())
    #plt.plot(test_error)
    #plt.show()
    submission.to_csv('sample_submssion_lgbm.csv')
    
    #save model by joblib library
    import joblib
    joblib.dump(models, 'saved_models.pkl')
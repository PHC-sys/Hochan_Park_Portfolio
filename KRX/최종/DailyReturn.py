from AnalyticUtility import *
import pandas as pd, FinanceDataReader as fdr
import time
import numpy as np
import pickle
import os

portfolio_data = pd.read_csv(r'C:\Users\John\OneDrive\바탕 화면\퀀트관련\KRX대회\최종\final_submission.csv')
portfolio_data = portfolio_data.sort_values(by='순위')
long_port = portfolio_data.head(200)['종목코드']
short_port = portfolio_data.tail(200)['종목코드']

print(long_port)

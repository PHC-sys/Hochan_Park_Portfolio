import pandas as pd
import os

read_path = '/Users/donghanko/Desktop/StockList/'
save_path = '/Users/donghanko/Desktop/SubmissionFile/'
try:
    os.mkdir(read_path)
    os.mkdir(save_path)
except:
    pass


# 폴더 내 모든 파일의 경로를 조회
all_files = [os.path.join(read_path, file) for file in os.listdir(read_path)]

# 파일들 중에서 마지막으로 생성된 파일을 조회
latest_file = max(all_files, key=os.path.getmtime)

data = pd.read_csv(latest_file,index_col=0)
df = pd.read_csv('/Users/donghanko/Downloads/open/sample_submission.csv')

result_df = pd.DataFrame(data.iloc[:,-1])
lst = []
for i in result_df.index:
    lst.append(f'A{i:06d}')

result_df.index = lst
result_df['순위'] = result_df[data.columns[-1]].rank(method='first', ascending=False).astype('int')
result_df['종목코드'] = result_df.index

baseline_submission = df[['종목코드']].merge(result_df[['종목코드', '순위']], on='종목코드', how='left')
print(baseline_submission)
baseline_submission.to_csv(save_path + 'baseline_submission.csv', index=False)
import pandas as pd
import datetime

df = pd.read_csv('dataset_11gb.csv') # data_sample_1mb.csv
# rows, cols = df.shape
print df.ExchangeBid.mean()
# for col in df.columns.tolist():
# 	# print col, len(df[col].unique().tolist())
# 	# print df[col].value_counts()
# 	if len(df[col].unique().tolist()) == rows:
# 		print col


# # print df.Outcome.unique().tolist()
# print df.Gender.value_counts()
# # print df.stack().value_counts()


# df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))


# # print df.Timestamp.tolist()
# df.to_csv('newSet.csv', headers=True, index=False)


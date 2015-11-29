import pandas as pd
import datetime

def appendColHeaders(givenDf):
	tempDf = pd.read_csv('data_sample_1mb.csv')
	givenDf.columns = tempDf.columns.tolist()
	return givenDf

df = pd.read_csv('ALL_DATASETS/dataset_11gb.csv', header=None)
mydf = appendColHeaders(df)
mydf.to_csv('dataset_11gb.csv', index=False)
print mydf.shape

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


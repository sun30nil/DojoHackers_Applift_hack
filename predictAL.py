# BidId   TrafficType     PublisherId     AppSiteId       AppSiteCategory
# Position        BidFloor        Timestamp       Age     Gender  OS
# OSVersion       Model   Manufacturer    Carrier DeviceType      DeviceId
# DeviceIP        Country Latitude        Longitude       Zipcode GeoType
# CampaignId      CreativeId      CreativeType    CreativeCategory
# ExchangeBid     Outcome

import pandas as pd
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn import preprocessing

# _dataFile = 'data_sample_1mb.csv'
# _dataFile = 'dataset_1gb.csv'
_dataFile = 'dataset_11gb.csv'

_paramList = ['TrafficType',
              'PublisherId',
              'AppSiteId',
              'AppSiteCategory',
              'Position',
              'BidFloor',
              'Age',
              'Gender',
              'OS',
              'OSVersion',
              'Carrier',
              'DeviceType',
              'Country',
              'Zipcode',
              'CampaignId',
              'CreativeId',
              'CreativeType',
              'CreativeCategory',
              'ExchangeBid',
              'Outcome'
              ]


def dropColumns(df_new, retainList):
    for eachCol in df_new.columns.tolist():
        if eachCol not in retainList:
            df_new.drop(eachCol, axis=1, inplace=True)
    return df_new


def transformColumns(df_new, cols_list):
    le = preprocessing.LabelEncoder()
    for col in cols_list:
        df_new[col] = le.fit_transform(df_new[col].values)
    return df_new

predict_col = 'Outcome'

testSize = 0.2
df_all = pd.read_csv(_dataFile)
df_all = dropColumns(df_all, _paramList)
df_all = transformColumns(df_all, df_all.columns.tolist()[:-1])
train_set, test_set = train_test_split(df_all, test_size=testSize)

print train_set.shape
print test_set.shape

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=len(_paramList)-1,
    learning_rate=0.5,
    algorithm="SAMME")

# bdt_discrete = GradientBoostingClassifier(n_estimators=len(_paramList) - 1, learning_rate=1.0,
#                                           max_depth=1, random_state=0)

y_train = train_set[predict_col].values
X_train = train_set.as_matrix(columns=train_set.columns[:-1])
X_test = test_set.as_matrix(columns=test_set.columns[:-1])
y_test = test_set[predict_col].values


bdt_discrete.fit(X_train, y_train)
print 'train : test', 1 - testSize, testSize
print 'Score :', bdt_discrete.score(X_test, y_test)


# predicted = bdt_discrete.predict(X_test)
# y_true = test_set[predict_col].values
# y_pred = np.array(predicted)
# print 'P - R - FS - S', precision_recall_fscore_support(y_true, y_pred, average='weighted')


# df_test['pred'] = predicted.ravel()
# df_test.to_csv("PRED.csv", index=False)

# print predicted

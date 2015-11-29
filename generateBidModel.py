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
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
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
              'DeviceIP',
              'Country',
              'Zipcode',
              'CampaignId',
              'CreativeId',
              'CreativeType',
              'CreativeCategory',
              'ExchangeBid'
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

predict_col = 'ExchangeBid'

testSize = 0.2
df_all = pd.read_csv(_dataFile)
df_all = dropColumns(df_all, _paramList)
df_all = transformColumns(df_all, df_all.columns.tolist()[:-1])
train_set = df_all


est = GradientBoostingRegressor(
    n_estimators=100, learning_rate=1, max_depth=1, random_state=0, loss='ls')

y_train = train_set[predict_col].values
X_train = train_set.as_matrix(columns=train_set.columns[:-1])

est.fit(X_train, y_train)
from sklearn.externals import joblib
joblib.dump(est, 'Models/bidModel_11g.pkl')

# print predicted

# predicted = bdt_discrete.predict(X_test)
# y_true = test_set[predict_col].values
# y_pred = np.array(predicted)
# print 'P - R - FS - S', precision_recall_fscore_support(y_true, y_pred,
# average='weighted')


# df_test['pred'] = predicted.ravel()
# df_test.to_csv("PRED.csv", index=False)

# print predicted

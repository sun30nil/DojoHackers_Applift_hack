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

_outcome_11g_model_path = 'Models/outcomemodel_11g.pkl'
_bid_11g_model_path = 'Models/bidModel_11g.pkl'


_paramListBid = ['TrafficType',
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
              'CreativeCategory'
              ]

_paramListOut = ['TrafficType',
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


def predictFunction(model_path, x_test):
  from sklearn.externals import joblib
  ld = joblib.load(model_path)
  predicted = ld.predict(x_test)
  return predicted


def returnPredicted(predict_col, dataFilePath):
  full_test_set = pd.read_csv(dataFilePath)
  test_set = pd.read_csv(dataFilePath)
  if predict_col == 'Outcome':
    test_set = dropColumns(test_set, _paramListOut)
    test_set = transformColumns(test_set, test_set.columns.tolist())
    X_test = test_set.as_matrix(columns=test_set.columns)
    y_test = predictFunction(_outcome_11g_model_path, X_test)
  elif predict_col == 'ExchangeBid':
    test_set = dropColumns(test_set, _paramListBid)
    test_set = transformColumns(test_set, test_set.columns.tolist())
    X_test = test_set.as_matrix(columns=test_set.columns)
    y_test = predictFunction(_bid_11g_model_path, X_test)
  full_test_set[predict_col] = y_test.ravel()
  full_test_set.to_csv("pred.csv", index=False)

  # print full_test_set.set_index('BidId')[predict_col].to_dict()

returnPredicted('Outcome', 'test_oc.csv')
# returnPredicted('ExchangeBid', 'test_eb.csv')
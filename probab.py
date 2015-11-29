# BidId   TrafficType     PublisherId     AppSiteId       AppSiteCategory
# Position        BidFloor        Timestamp       Age     Gender  OS
# OSVersion       Model   Manufacturer    Carrier DeviceType      DeviceId
# DeviceIP        Country Latitude        Longitude       Zipcode GeoType
# CampaignId      CreativeId      CreativeType    CreativeCategory
# ExchangeBid     Outcome

import pandas as pd
import datetime
import operator
import sys

df = pd.read_csv(sys.argv[1])

# print len(df.DeviceIP.unique().tolist())
target_col = sys.argv[2]

map_of_interest = df[target_col].value_counts().to_dict()

userList = []
prob_o = []
prob_w = []
prob_c = []
for k, v in map_of_interest.iteritems():
	userList.append(k)
	tempDf1 = df[df[target_col] == k]
	max_val = v
	tempDf_W = tempDf1[tempDf1.Outcome == 'w']
	prob_w.append(tempDf_W.shape[0]/float(max_val))

	tempDf_0 = tempDf1[tempDf1.Outcome == '0']
	prob_o.append(tempDf_0.shape[0]/float(max_val))

	tempDf_C = tempDf1[tempDf1.Outcome == 'c']
	prob_c.append(tempDf_C.shape[0]/float(max_val))

col_headers = [target_col, 'w_prob', '0_prob', 'c_prob']
new_df = pd.DataFrame(columns=col_headers)

new_df[target_col] = userList
new_df['w_prob'] = prob_w
new_df['0_prob'] = prob_o
new_df['c_prob'] = prob_c

new_df.to_csv("prob.csv", index=False)

gist = new_df.sort(['w_prob'], ascending=False)
print gist.head(5)





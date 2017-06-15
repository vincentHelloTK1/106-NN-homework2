import numpy as np
import sys
import os
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,LinearSVR,NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import xgboost as xgb


np.set_printoptions(suppress=True)
##############
filepath=sys.argv[1]
f = open(filepath)
dataset = np.loadtxt(f,delimiter=',',skiprows=1)
target=dataset[:,0]
data_train=dataset[:,1:]

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
data_train = scaler.fit_transform(data_train)

## traing  ######################################################################

#regr=linear_model.Lasso()
#regr=linear_model.BayesianRidge()
#regr=linear_model.BayesianRidge()
#regr=linear_model.BayesianRidge(normalize=True)
#regr=linear_model.ElasticNet()
#regr = xgb.XGBRegressor(max_depth=2,n_estimators=300,learning_rate=0.01)
#regr = MLPRegressor(hidden_layer_sizes=(200, ))
#regr = linear_model.BayesianRidge()
#regr = RandomForestRegressor(max_depth=2,n_estimators=100)
#regr = linear_model.Lasso()
#regr = RandomForestRegressor(max_depth=5,n_estimators=100)
regr = xgb.XGBRegressor(max_depth=2,n_estimators=300,learning_rate=0.01)
#regr = SVR(C=10,degree=2)

regr.fit(data_train,target)
#print regr.score(data_train,target)


## predict ########################################################

####
filepath=sys.argv[2]
f = open(filepath)
dataset2 = np.loadtxt(f,delimiter=',',skiprows=1)
target2=dataset2[:,0]
data_test=dataset2[:,1:]

data_test = scaler.fit_transform(data_test)


listpredict=list()
result=regr.predict(data_test)

###
i=0
f=open('testuniqid')
print 'intersection_id,tollgate_id,time_window,avg_travel_time'
for line in f :
  tmp=line.split(';')
  print tmp[0]+','+tmp[1]+','+tmp[2].strip()+','+str(round(result[i],2))
  i=i+1

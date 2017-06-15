import pandas as pd
import numpy as np
import math
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn import preprocessing
from datetime import datetime,timedelta

def avgTravelTime(train,test):
	dic = {'A':1,'B':2,'C':3}
	for i in range(len(train['intersection_id'])):
	    train.loc[i,'intersection_id'] = dic[train['intersection_id'][i]]
	for i in range(len(test['intersection_id'])):
	    test.loc[i,'intersection_id'] = dic[test['intersection_id'][i]]
	temp = np.array(train['sea_pressure'])
	for i in range(len(temp)):
	    if (math.isnan(float(temp[i]))):
	        train = train.drop(i)
	temp = np.array(test['sea_pressure'])
	for i in range(len(temp)):
	    if (math.isnan(float(temp[i]))):
	        test = test.drop(i)
	train_y = np.array(train.avg_travel_time)
	train_x = np.array(train.drop('avg_travel_time',axis = 1).drop('time_window',axis = 1))
	train_x = preprocessing.scale(train_x)#Normalization

	X_train, X_test, y_train, y_test = train_test_split(train_x, train_y,test_size=0.1, random_state=0)
	linreg = MLPRegressor(hidden_layer_sizes=(100,200,80,50,200,300,60,200,200,200,200,200,250,280,209,207,250,220,240,240,240,240,240,240,240,240,240,240,240,500,99,455,351,281,216,324,150,222,546,425,654,156),learning_rate='adaptive',max_iter=10000)
	linreg.fit(X_train, y_train)
	y_pred = linreg.predict(X_test)
	print "MAE: ",metrics.mean_absolute_error(y_test, y_pred)
	print "RMSE: ",np.sqrt(metrics.mean_squared_error(y_test, y_pred))

	test_x = np.array(test.drop('avg_travel_time',axis = 1).drop('time_window',axis = 1))
	test_y = linreg.predict(test_x)

	for i in range(len(test.time_window)):
	    test.loc[i,'avg_travel_time'] = round(test_y[i]/9,2)

	test.to_csv('20min_avg_travel_time.csv')

def main():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    avgTravelTime(train,test)

if __name__ == '__main__':
    main()

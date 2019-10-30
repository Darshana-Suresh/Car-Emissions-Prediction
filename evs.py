# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:39:32 2019

@author: sojan,dashy,shalini,saai
"""

import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

dataframe=pandas.read_csv('evs.csv', encoding='windows-1252')

"""
PRE PROCESSING---------------------------------------------------------
"""
dataframe.drop('elecenergycons',axis=1,inplace=True)
dataframe.drop('Electricity cost',axis=1,inplace=True)
dataframe.drop('wh/km',axis=1,inplace=True)
dataframe.drop('Maximumrange(Miles)',axis=1,inplace=True)
dataframe.drop('Maximumrange(Km)',axis=1,inplace=True)
dataframe.drop('Manufacturer',axis=1,inplace=True)
dataframe.drop('Transmission',axis=1,inplace=True)
dataframe.drop('Model',axis=1,inplace=True)
dataframe.drop('Description',axis=1,inplace=True)

var=0
darray=dataframe['MetricUrban (Cold)'].values
dataframe['MetricUrban (Cold)'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['THCEmissions[mg/km]'].fillna(var,inplace=True)
var=0
darray=dataframe['THCEmissions[mg/km]'].values
dataframe['THCEmissions[mg/km]'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['EngineCapacity'].fillna(var,inplace=True)
var=0
darray=dataframe['EngineCapacity'].values
dataframe['EngineCapacity'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['EngineCapacity'].fillna(var,inplace=True)
var=0
darray=dataframe['Metric Extra-Urban'].values
dataframe['Metric Extra-Urban'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['Metric Extra-Urban'].fillna(var,inplace=True)
var=0
darray=dataframe['Metric Combined'].values
dataframe['Metric Combined'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['Metric Combined'].fillna(var,inplace=True)
var=0
darray=dataframe['Imperial Urban (Cold)'].values
dataframe['Imperial Urban (Cold)'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['Imperial Urban (Cold)'].fillna(var,inplace=True)
var=0
darray=dataframe['Imperial Extra-Urban'].values
dataframe['Imperial Extra-Urban'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['Imperial Extra-Urban'].fillna(var,inplace=True)
var=0
darray=dataframe['Imperial Combined'].values
dataframe['Imperial Combined'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['Imperial Combined'].fillna(var,inplace=True)
var=0
darray=dataframe['EuroStandard'].values
dataframe['EuroStandard'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['EuroStandard'].fillna(var,inplace=True)
var=0
darray=dataframe['Emissions CO [mg/km]'].values
dataframe['Emissions CO [mg/km]'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['Emissions CO [mg/km]'].fillna(var,inplace=True)
var=0
darray=dataframe['NoiseLeveldB(A)'].values
dataframe['NoiseLeveldB(A)'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['NoiseLeveldB(A)'].fillna(var,inplace=True)
var=0
darray=dataframe['Particulates[No.][mg/km]'].values
dataframe['Particulates[No.][mg/km]'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['Particulates[No.][mg/km]'].fillna(var,inplace=True)

var=0
darray=dataframe['EmissionsNOx[mg/km]'].values
dataframe['EmissionsNOx[mg/km]'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['EmissionsNOx[mg/km]'].fillna(var,inplace=True)

var=0
darray=dataframe['THC+NOxEmissions [mg/km]'].values
dataframe['THC+NOxEmissions [mg/km]'].fillna(0,inplace=True)
for i in range(5118):
    var+=darray[i]
var/=5118
dataframe['THC+NOxEmissions [mg/km]'].fillna(var,inplace=True)

"""
Linear Regression -----------------------------------------------------------------------------
"""

array=dataframe.values
X=array[:,[0,2,3,4,5,6,7,8,11,12,14,17]]
Y=array[:,[16]]
x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.2, random_state=0)
 # qsec is the time taken to  travel quatermile
model=LinearRegression()

results=model.fit(x_train,y_train)

print("the coefficients are")
print(results.coef_)
print()
print("the intercept is")
print(results.intercept_)    
print()
print("score is")
print(round(results.score(X,Y),5))
print("%.3f%%" % (results.score(X,Y)*100.0))
print()


predicted=model.predict(x_test)
y_predicted=0
for i in range(y_test.shape[0]):
    y_predicted+=(y_test[i]-predicted[i])**2
    predicted[i]=y_test[i]-predicted[i]

y_predicted=y_predicted/y_test.shape[0]
print()
print( "RMS VALUE:%.3f" % y_predicted**(0.5)) # ** means power
"""

"""

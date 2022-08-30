import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
%matplotlib inline

#import dataset from sklearn.dataset and print it

from sklearn.datasets import load_boston
boston=load_boston()
print(boston)

# transfrom dataset to data-frame
# data = data we want// independent variables also known as the x values
#feature_name = the column names of the data
#target = target variable (price//y value)
df_x=pd.DataFrame(boston.data,columns=boston.feature_names)
df_y=pd.DataFrame(boston.target)


#get some stats from the dataset , count,mean
df_x.describe()

#initialise thde linear regression
reg=linear_model.LinearRegression()

#split data in train 67% and test 33%
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.33,random_state=42)

#train model with data
reg.fit(x_train,y_train)

#print coeff//weight for each feature // column for our model
print(reg.coef_) #f(x,a)=mx + da + b =y

#print prediction on dataset 
y_pred=reg.predict(x_test)
print(y_pred)

#print actual values
print(y_test)

#check model efficiency//performance using Mean Squared Error(MSE)
print(np.mean((y_pred - y_test)**2))

#check model efficiency//performance using Mean Squared Error(MSE) and sk.learn.metrics
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))
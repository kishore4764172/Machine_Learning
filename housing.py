import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing

#load the data from the datasets
housing=fetch_california_housing()

#create a table
df=pd.DataFrame(housing.data,columns=housing.feature_names)
df["target"]=housing.target

#drops the target column
x=df.drop("target",axis=1)
y=df["target"]   # create a column y label

#splits the data into training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#trains the model
linear=LinearRegression()
linear.fit(x_train,y_train)

#evaluate
y_pre=linear.predict(x_test)
me=mean_squared_error(y_test,y_pre)
r2=r2_score(y_test,y_pre)

print("meansquare is : ",me)
print("r2 : ",r2)











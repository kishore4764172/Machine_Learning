import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


hours={
    "studied_hours":[1,2,3,4.5,5,6,7.5,8,9,10],
    "score":[10,20,30,40,50,60,70,80,85,95]
}

df=pd.DataFrame(hours)
print(df)

plt.scatter(df["studied_hours"],df["score"],color="blue")
plt.title("Student_score")
plt.xlabel("input feature")
plt.ylabel("target value")
plt.grid()
plt.show()

x=df[["studied_hours"]]
y=df["score"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

pre_score=LinearRegression()
pre_score.fit(x_train,y_train)


print(f"Intercept : {pre_score.intercept_}")
print(f"Coefficient : {pre_score.coef_}")
y_pre=pre_score.predict(x_test)
ms=mean_squared_error(y_test,y_pre)
r2=r2_score(y_test,y_pre)

line=pre_score.coef_* x + pre_score.intercept_

plt.scatter(x,y,color="blue")
plt.plot(x,line,color="red")
plt.title("regression line")
plt.xlabel("studied_hours")
plt.ylabel("score")
plt.grid()
plt.show()

hours=[[9]]
predicted_score=pre_score.predict(hours)
print(f"The predict value for 9 hours is : {predicted_score[0]}")


print("ms value : ",ms)
print("r2 value : ",r2)


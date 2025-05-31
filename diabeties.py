import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#load the data
df=pd.read_csv("diabetes.csv")

x=df.drop("Outcome",axis=1)
y=df["Outcome"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#trains the model
model=RandomForestClassifier()
model.fit(x_train,y_train)

#predicts the data
y_pre=model.predict(x_test)

#evaluate
predict=accuracy_score(y_test,y_pre)
print("accuracy :",predict)




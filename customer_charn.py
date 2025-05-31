import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report

#reads the csv
df=pd.read_csv("Telco-Customer-Churn.csv")

df.drop("customerID",axis=1,inplace=True)
df.dropna(inplace=True)

le=LabelEncoder()
for column in df.columns:
    if df[column].dtype==object:
        df[column]=le.fit_transform(df[column])

x=df.drop("Churn",axis=1)
y=df["Churn"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(x_train,y_train)

y_prd=model.predict(x_test)
accur=accuracy_score(y_test,y_prd)
Classification=classification_report(y_test,y_prd)
print("accuracy : ",accur)
print("\nClassification Report :\n",Classification)





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#loads the datasets
iris=load_iris()

#create the DataFrame
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df["target"]=iris.target


#input feature and target
x=df.drop("target",axis=1)
y=df["target"]

#trains and tests the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#scales the data
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#trains the model
KNN=KNeighborsClassifier()
KNN.fit(x_train,y_train)

#predicts the model
y_predict=KNN.predict(x_test)

#evaluate the model
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)

import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#load the iris data from the datasets
iris=datasets.load_iris()

random=RandomForestClassifier()

#creates a table
df=pd.DataFrame(iris.data,columns=iris.feature_names)

#creates a new column
df["target"]=iris.target

#creats a new columns and maps the species from the species
df["species"]=df["target"].map({i:species for i,species in enumerate(iris.target_names)})

x=df[iris.feature_names]
y=df["species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print("x_train : ",x_train.shape)
print("x_test : ",x_test.shape)

print("y_train : ",y_train.shape)
print("y_test : ",y_test.shape)
print()

random.fit(x_train,y_train)

y_pre=random.predict(x_test)
print(y_pre)
print("accuracy : ",accuracy_score(y_test,y_pre))
print()
"""
#virtualize the image
sns.pairplot(df,hue="species")
plt.show()
"""

print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#load the data
digits=load_digits()

#input feature and labels
x=digits.data
y=digits.target

#trains and tests the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#trains the model
neighbors=KNeighborsClassifier()
neighbors.fit(x_train,y_train)

#predicts the model
y_prd=neighbors.predict(x_test)

#evaluates the model
print("accuracy : ",accuracy_score(y_test,y_prd))

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    plt.title(f"prediction : {neighbors.predict([digits.data[i]])[0]}")
    plt.show()
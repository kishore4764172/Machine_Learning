import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

text={
    'data': [
        "Congratulations! You've won a $1000 gift card.",
        "Hi John, can we reschedule the meeting?",
        "Lowest prices on your favorite brands. Click here!",
        "Reminder: Your appointment is tomorrow at 10 AM.",
        "You have been selected for a prize. Claim now!",
        "Hey, just checking in. How are you?"
    ],
    'label': [1, 0, 1, 0, 1, 0]
}

df=pd.DataFrame(text)


vectorizer=CountVectorizer()

x=vectorizer.fit_transform(df["data"])
y=df["label"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(x_train,y_train)

y_pre=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pre)

print("accuracy : ",accuracy)
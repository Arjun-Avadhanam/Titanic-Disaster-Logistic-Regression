import streamlit as st
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
def load_data():
    #"""train data processing"""
    data = pd.read_csv("train.csv")
    data = data.drop(['Ticket','PassengerId',"Name","Cabin"],axis=1)
    #data['Embarked'] = data['Embarked'].fillna('S')
    # med_fare = data.groupby(['Pclass', 'Parch', 'SibSp'],group_keys=True).Fare.median()[3][0][0]
    # data['Fare'] = data['Fare'].fillna(med_fare)
    # data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    data["Embarked"].replace(["Q","S","C"],[0,1,2],inplace=True)
    data["Sex"].replace(["male","female"],[0,1],inplace=True)
    normalised_data = (data - data.min())/(data.max()- data.min())
    data = normalised_data.sample(frac=1).reset_index(drop=True)
    data = data.fillna(0)
    return data

def load_test_data_match():
    test_data = pd.read_csv("test.csv")
    test_data_match = test_data.drop(['Ticket',"Name","Cabin"],axis=1)
    return test_data_match


def load_test_data():
    test_data = pd.read_csv("test.csv")
    test_data = test_data.drop(['Ticket','PassengerId',"Name","Cabin"],axis=1) 
    test_data["Embarked"].replace(["Q","S","C"],[0,1,2],inplace=True)
    test_data["Sex"].replace(["male","female"],[0,1],inplace=True)
    normalised_test_data = (test_data - test_data.min())/(test_data.max()- test_data.min())
    test_data = normalised_test_data.fillna(0)
    return test_data
test_data_matcher = load_test_data_match()

test_data_df = load_test_data()

display_data = load_data()




validation_ratio =0.2

train,validated_data = train_test_split(display_data,test_size=validation_ratio)

train_df = pd.DataFrame(train)
train_df = train_df.drop(["Survived"],axis=1)
train_survived = train["Survived"]

val_df = pd.DataFrame(validated_data)
val_df = val_df.drop(["Survived"],axis=1)
val_survived = validated_data["Survived"]

Log_Reg = LogisticRegression()
Log_Reg.fit(train_df, train_survived)

prediction = Log_Reg.predict(val_df)
test_prediction = pd.DataFrame(prediction)


test_pred = Log_Reg.predict(test_data_df)

result = pd.DataFrame(test_data_matcher["PassengerId"])
result.insert(1, "Survived", test_pred, True)
result["Survived"] = pd.to_numeric(result["Survived"], downcast="integer")


result.to_csv("output3.csv", index=False)
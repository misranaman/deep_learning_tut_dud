import pickle

import pandas
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

st.title("Passenger Survival Chance In Titanic Journey")

p_class = st.slider("Enter The Passenger Class For The User", 1, 3)
sex = st.selectbox("Enter the Passenger Gender", ["male", "female"])
sib_sp = st.slider("Enter the Passenger Sibling and Spouse", 1, 8)
par_ch = st.slider("Enter the Passenger total number of Parents and Child", 0, 6)
fare = st.number_input("Enter the Fare of The Passenger")
embarked = st.selectbox("Enter the Passenger starting journey", ["Southampton", "Chebourg", "Queenstown"])

data = pd.DataFrame([
    {"Pclass": p_class, "Sex": sex, "SibSp": sib_sp, "Parch": par_ch, "Fare": fare, "Embarked": embarked}])
# if st.button("Data"):
#     st.write(data)

model = load_model("model.h5")
with open("label_encoder.pkl", "rb") as file:
    label: LabelEncoder = pickle.load(file);

with open("one_hot_encoder.pkl", "rb") as file:
    one_hot: OneHotEncoder = pickle.load(file)

with open("scalar.pkl", "rb") as file:
    scalar: StandardScaler = pickle.load(file)
data["Sex"] = label.transform(data["Sex"])
embarked = one_hot.transform(data[["Embarked"]])

embarked = pd.DataFrame(embarked, columns=one_hot.get_feature_names_out())
data = pandas.concat([data.drop(columns=["Embarked"]), embarked], axis=1)
data[["Pclass", "SibSp", "Parch", "Fare"]] = scalar.transform(data[["Pclass", "SibSp", "Parch", "Fare"]])

y = model.predict(data)

y = y[0][0]


def chance(predicted):
    if predicted > 0.5:
        st.write("Passenger will survive the journey")
    else:
        st.write("Passenger will not survive the journey")


if st.button("Predict"):
    st.write("Probability of passengers survival", y)
    chance(y)

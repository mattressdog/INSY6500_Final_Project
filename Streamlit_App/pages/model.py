import streamlit as st
import pandas as pd
from model_core import read_data, logistic_regression, KNN, SVM

questions = ["Logistic Regression", "KNN", "SVM"]

st.title("Analysis")
st.write("This page contains Predictive Analysis Models to determine which columns contribute most to the survivability of patients.")
st.write("For more information, view the presentation and notebook associated with this project.")

question = st.selectbox("Select the model for which to run the analysis:", questions)

run = st.button("Run Analysis")

if run:
    if question == "Logistic Regression":
        st.write(logistic_regression())
    elif question == "KNN":
        st.write(KNN())
    elif question == "SVM":
        st.write(SVM())

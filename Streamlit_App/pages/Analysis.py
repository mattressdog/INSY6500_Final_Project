import streamlit as st
import pandas as pd
from core import read_data , create_bar_chart, create_kde_chart, Question1, Question2, Question3

questions = ["Question 1", "Question 2", "Question 3"]

st.title("Analysis")
st.write("This page contains analysis charts from the INSY6500 Final Project")
st.write("For more information, view the presentation and notebook associated with this project.")

question = st.selectbox("Select the column to view the analysis charts for:", questions)

if question == "Question 1":
    st.write(Question1())
elif question == "Question 2":
    st.write(Question2())
elif question == "Question 3":
    st.write(Question3())

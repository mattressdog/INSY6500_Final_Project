#INSY6500 Final Project Streamlit App
#Some code in this app was AI generated, but all code was modified to fit the data and output desired in this app. 
#AI intellisense was used to generate code within this app.
import streamlit as st
from core import read_data , create_bar_chart, create_kde_chart

st.set_page_config(page_title="INSY6500 Final Project", layout="wide")

st.title("INSY6500 Final Project Fall 2025")
st.write("Breast Cancer Research Project")

#Streamlit App
st.write("Authors: Caitlin Tran and  Matthew Ross")

#Data
df = read_data()
st.write("Data for experimental research purposes only.")
st.write(df)
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
small_unique_cols = [col for col in df.columns if df[col].nunique() < 10]

x_col = st.selectbox("Select the column to view the bar chart for:", small_unique_cols)

#if st.button("View Data"):
chart_1 = create_bar_chart(df[x_col])
st.write(chart_1)

#st.write("This is a KDE chart")
choice = st.selectbox("Select a column to view the KDE chart for:", numeric_cols)
#if st.button("View Data"):
chart_2 = create_kde_chart(df[choice])
st.write(chart_2)


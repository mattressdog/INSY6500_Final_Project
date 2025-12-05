import matplotlib.pylab as plt
import streamlit as st
import pandas as pd
import sys
import numpy as np
import sklearn
import scipy
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
data_root = "https://raw.githubusercontent.com/mattressdog/data/refs/heads/main/"


def read_data():
    df = pd.read_csv(data_root + "Breast_Cancer_Primary.csv",
                    dtype={"Race": "string",
                            "Marital Status": "string",
                            "T Stage": "string",
                            "N Stage": "string",
                            "6th Stage": "string",
                            "differentiate": "string",
                            #"Grade": "int64",
                            "A Stage": "string",
                            #"ride_id": "string",
                            "Status": "string",})
    df.columns = df.columns.str.strip()
    return df

def logistic_regression():
    df = read_data()
    df["A_Stage_Binary"] = df["A Stage"].map({"Regional": 1, "Distant": 0})
    df["Estrogen_Binary"] = df["Estrogen Status"].map({"Positive": 1, "Negative": 0})
    df["Progesterone_Binary"] = df["Progesterone Status"].map({"Positive": 1, "Negative": 0})
    df = df.drop(columns=["A Stage"])
    df = df.drop(columns=["Estrogen Status"])
    df = df.drop(columns=["Progesterone Status"])
    df = df.drop(columns=["Grade"])
    df = df.drop(columns=["Regional Node Examined"])
    df = df.drop(columns=["Reginol Node Positive"])
    X = df.drop(columns=["Status"])
    y = df["Status"]
    #This dataset contains categorical and ordinal values that need to be included in the analysis
    categorical_cols = ["Marital Status", "Race"]
    ordinal_cols = ["T Stage", "N Stage", "6th Stage", "differentiate"]
    ordinal_order = [["T1", "T2", "T3","T4"],["N1", "N2", "N3"],["IIA", "IIB", "IIIA", "IIIB", "IIIC"],["Poorly differentiated", "Moderately differentiated", "Well differentiated","Undifferentiated"]]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop='first'), categorical_cols),
            ("ord", OneHotEncoder(categories=ordinal_order, drop='first'), ordinal_cols)
        ],
        remainder="passthrough"
    )

    X_processed = preprocess.fit_transform(X)
    ohe_cat = preprocess.named_transformers_["cat"]
    ohe_ord = preprocess.named_transformers_["ord"]

    cat_names = ohe_cat.get_feature_names_out(["Marital Status", "Race"])
    ord_names = ohe_ord.get_feature_names_out(["T Stage", "N Stage", "6th Stage", "differentiate"])

    # Numeric columns passed through unchanged
    numeric_names = [
        col for col in X.columns 
        if col not in ["Marital Status", "Race", "T Stage", "N Stage", "6th Stage", "differentiate"]
    ]

    feature_names = np.concatenate([cat_names, ord_names, numeric_names])
    X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
    )

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    logreg_pred = logreg.predict(X_test)

    logreg_pi = permutation_importance(
        logreg, X_test, y_test, n_repeats=20, random_state=42
    )

    logreg_importances = logreg_pi.importances_mean

    st.write(classification_report(y_test, logreg_pred))

    st.write("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))

    df_logreg_pi = pd.DataFrame({
        "feature": feature_names,
        "importance": logreg_importances
    }).sort_values("importance", ascending=False)

    st.write(df_logreg_pi)

    return 0

def KNN():
    df = read_data()
    df["A_Stage_Binary"] = df["A Stage"].map({"Regional": 1, "Distant": 0})
    df["Estrogen_Binary"] = df["Estrogen Status"].map({"Positive": 1, "Negative": 0})
    df["Progesterone_Binary"] = df["Progesterone Status"].map({"Positive": 1, "Negative": 0})
    df = df.drop(columns=["A Stage"])
    df = df.drop(columns=["Estrogen Status"])
    df = df.drop(columns=["Progesterone Status"])
    df = df.drop(columns=["Grade"])
    df = df.drop(columns=["Regional Node Examined"])
    df = df.drop(columns=["Reginol Node Positive"])
    X = df.drop(columns=["Status"])
    y = df["Status"]
    #This dataset contains categorical and ordinal values that need to be included in the analysis
    categorical_cols = ["Marital Status", "Race"]
    ordinal_cols = ["T Stage", "N Stage", "6th Stage", "differentiate"]
    ordinal_order = [["T1", "T2", "T3","T4"],["N1", "N2", "N3"],["IIA", "IIB", "IIIA", "IIIB", "IIIC"],["Poorly differentiated", "Moderately differentiated", "Well differentiated","Undifferentiated"]]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop='first'), categorical_cols),
            ("ord", OneHotEncoder(categories=ordinal_order, drop='first'), ordinal_cols)
        ],
        remainder="passthrough"
    )

    X_processed = preprocess.fit_transform(X)
    ohe_cat = preprocess.named_transformers_["cat"]
    ohe_ord = preprocess.named_transformers_["ord"]

    cat_names = ohe_cat.get_feature_names_out(["Marital Status", "Race"])
    ord_names = ohe_ord.get_feature_names_out(["T Stage", "N Stage", "6th Stage", "differentiate"])

    # Numeric columns passed through unchanged
    numeric_names = [
        col for col in X.columns 
        if col not in ["Marital Status", "Race", "T Stage", "N Stage", "6th Stage", "differentiate"]
    ]

    feature_names = np.concatenate([cat_names, ord_names, numeric_names])
    X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    knn_pred = knn.predict(X_test)

    knn_pi = permutation_importance(
        knn, X_test, y_test, n_repeats=20, random_state=42
    )

    knn_importances = knn_pi.importances_mean
    st.write("KNN:", accuracy_score(y_test, knn_pred))

    df_knn_pi = pd.DataFrame({
        "feature": feature_names,
        "importance": knn_importances
    }).sort_values("importance", ascending=False)

    st.write(df_knn_pi)

    return 0

def SVM():
    df = read_data()
    df["A_Stage_Binary"] = df["A Stage"].map({"Regional": 1, "Distant": 0})
    df["Estrogen_Binary"] = df["Estrogen Status"].map({"Positive": 1, "Negative": 0})
    df["Progesterone_Binary"] = df["Progesterone Status"].map({"Positive": 1, "Negative": 0})
    df = df.drop(columns=["A Stage"])
    df = df.drop(columns=["Estrogen Status"])
    df = df.drop(columns=["Progesterone Status"])
    df = df.drop(columns=["Grade"])
    df = df.drop(columns=["Regional Node Examined"])
    df = df.drop(columns=["Reginol Node Positive"])
    X = df.drop(columns=["Status"])
    y = df["Status"]
    #This dataset contains categorical and ordinal values that need to be included in the analysis
    categorical_cols = ["Marital Status", "Race"]
    ordinal_cols = ["T Stage", "N Stage", "6th Stage", "differentiate"]
    ordinal_order = [["T1", "T2", "T3","T4"],["N1", "N2", "N3"],["IIA", "IIB", "IIIA", "IIIB", "IIIC"],["Poorly differentiated", "Moderately differentiated", "Well differentiated","Undifferentiated"]]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop='first'), categorical_cols),
            ("ord", OneHotEncoder(categories=ordinal_order, drop='first'), ordinal_cols)
        ],
        remainder="passthrough"
    )

    X_processed = preprocess.fit_transform(X)
    ohe_cat = preprocess.named_transformers_["cat"]
    ohe_ord = preprocess.named_transformers_["ord"]

    cat_names = ohe_cat.get_feature_names_out(["Marital Status", "Race"])
    ord_names = ohe_ord.get_feature_names_out(["T Stage", "N Stage", "6th Stage", "differentiate"])

    # Numeric columns passed through unchanged
    numeric_names = [
        col for col in X.columns 
        if col not in ["Marital Status", "Race", "T Stage", "N Stage", "6th Stage", "differentiate"]
    ]

    feature_names = np.concatenate([cat_names, ord_names, numeric_names])
    X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
    )

    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)

    svm_pred = svm.predict(X_test)

    svm_pi = permutation_importance(
        svm, X_test, y_test, n_repeats=20, random_state=42
    )

    svm_importances = svm_pi.importances_mean
    st.write("SVM:", accuracy_score(y_test, svm_pred))

    df_svm_pi = pd.DataFrame({
        "feature": feature_names,
        "importance": svm_importances
    }).sort_values("importance", ascending=False)

    st.write(df_svm_pi)

    return 0
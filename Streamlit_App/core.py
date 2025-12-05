import matplotlib.pylab as plt
import streamlit as st
import pandas as pd
import sys
import numpy as np
import sklearn
import scipy
import matplotlib
import seaborn as sns
from scipy.stats import gaussian_kde

def read_data():
    data_root = "https://raw.githubusercontent.com/mattressdog/data/refs/heads/main/"
    #Data Cleaning
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

def create_bar_chart(data):
    fig, ax = plt.subplots()
    sns.barplot(data=data)
    plt.show()
    return fig

def create_kde_chart(data):
    fig, ax = plt.subplots()
    sns.kdeplot(data)
    plt.show()
    return fig

def Question1():
    df = read_data()
    #Figure 3 TODO: Figures need titles and axes
    df['Age'].value_counts().sort_index().plot(kind='bar')
    plt.title("Counts of 'Age'")
    plt.ylabel("Count")
    plt.xlabel("Age")
    st.pyplot(plt)

    df_over_45 = df[df["Age"] > 45]
    df_under_45 = df[df["Age"] <= 45]

    counts = df_over_45['Status'].value_counts().sort_index()
    nums = counts.plot(kind='bar')
    plt.bar_label(nums.containers[0], padding=3)
    plt.title("Counts of 'Status' Ages 46 and Up")
    plt.ylabel("Count")
    plt.xlabel("Status")
    st.pyplot(plt)

    counts = df_under_45['Status'].value_counts().sort_index()
    nums = counts.plot(kind='bar')
    plt.bar_label(nums.containers[0], padding=3)
    plt.title("Counts of 'Status' Ages 45 and Under")
    plt.ylabel("Count")
    plt.xlabel("Status")
    st.pyplot(plt)

    st.title("Tumor Detection")

    sns.histplot(df_over_45["Tumor Size"])
    plt.title("Tumor Size Distribution for Patients 46 and Older")
    plt.xlabel("Tumor Size")
    plt.ylabel("Count")
    st.pyplot(plt)

    sns.histplot(df_under_45["Tumor Size"])
    plt.title("Tumor Size Distribution for Patients 45 and Younger")
    plt.xlabel("Tumor Size")
    plt.ylabel("Count")
    st.pyplot(plt)

    plt.figure(figsize=(8,5))

    sns.kdeplot(
        data=df_over_45["Tumor Size"],
        label="Age < 45",
        fill=True
    )

    sns.kdeplot(
        data=df_under_45["Tumor Size"],
        label="Age ≥ 45",
        fill=True
    )

    plt.title("KDE Plot of Tumor Size by Age Group")
    plt.xlabel("Tumor Size")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)

    x1 = df_over_45["Tumor Size"]
    x2 = df_under_45["Tumor Size"]

    # Fit KDEs manually (needed to compute intersections)
    kde1 = gaussian_kde(x1)
    kde2 = gaussian_kde(x2)

    # Shared x-grid
    xs = np.linspace(min(x1.min(), x2.min()),
                    max(x1.max(), x2.max()),
                    500)

    y1 = kde1(xs)
    y2 = kde2(xs)

    # Find intersection points
    diff = y1 - y2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    intersection_x = xs[sign_changes]

    # Plot curves
    plt.figure(figsize=(8,5))
    sns.kdeplot(x1, label='46 and Older')
    sns.kdeplot(x2, label='45 and Under')

    # Add vertical lines at intersections
    for x in intersection_x:
        plt.axvline(x, color='black', linestyle='--', alpha=0.7)

    plt.title("KDE Plot with Tumor Size by Age Group Intersection Points")
    plt.legend()
    st.pyplot(plt)

    st.write("Intersection points:", intersection_x)

    st.title("Cancer Stage and Survivability")

    N1_over_45 = df_over_45[(df_over_45['N Stage'] == 'N1')]
    N2_over_45 = df_over_45[(df_over_45['N Stage'] == 'N2')]
    N3_over_45 = df_over_45[(df_over_45['N Stage'] == 'N3')]
    N1_under_45 = df_under_45[(df_under_45['N Stage'] == 'N1')]
    N2_under_45 = df_under_45[(df_under_45['N Stage'] == 'N2')]
    N3_under_45 = df_under_45[(df_under_45['N Stage'] == 'N3')]

    s1 = N1_over_45['Status'].value_counts(normalize=True).sort_index()
    s3 = N2_over_45['Status'].value_counts(normalize=True).sort_index()
    s5 = N3_over_45['Status'].value_counts(normalize=True).sort_index()
    s2 = N1_under_45['Status'].value_counts(normalize=True).sort_index()
    s4 = N2_under_45['Status'].value_counts(normalize=True).sort_index()
    s6 = N3_under_45['Status'].value_counts(normalize=True).sort_index()


    combined_N = pd.DataFrame({'N1_over_45': s1, 'N1_under_45': s2, 'N2_over_45': s3, 'N2_under_45': s4, 'N3_over_45': s5, 'N3_under_45': s6})
    combined_N.plot(kind='bar')
    plt.title("Age vs N Stage")
    plt.ylabel("Count")
    st.pyplot(plt)

    T1_over_45 = df_over_45[(df_over_45['T Stage'] == 'T1')]
    T2_over_45 = df_over_45[(df_over_45['T Stage'] == 'T2')]
    T3_over_45 = df_over_45[(df_over_45['T Stage'] == 'T3')]
    T4_over_45 = df_over_45[(df_over_45['T Stage'] == 'T4')]
    T1_under_45 = df_under_45[(df_under_45['T Stage'] == 'T1')]
    T2_under_45 = df_under_45[(df_under_45['T Stage'] == 'T2')]
    T3_under_45 = df_under_45[(df_under_45['T Stage'] == 'T3')]
    T4_under_45 = df_under_45[(df_under_45['T Stage'] == 'T4')]

    s1 = T1_over_45['Status'].value_counts(normalize=True).sort_index()
    s3 = T2_over_45['Status'].value_counts(normalize=True).sort_index()
    s5 = T3_over_45['Status'].value_counts(normalize=True).sort_index()
    s7 = T4_over_45['Status'].value_counts(normalize=True).sort_index()
    s2 = T1_under_45['Status'].value_counts(normalize=True).sort_index()
    s4 = T2_under_45['Status'].value_counts(normalize=True).sort_index()
    s6 = T3_under_45['Status'].value_counts(normalize=True).sort_index()
    s8 = T4_under_45['Status'].value_counts(normalize=True).sort_index()



    combined_T = pd.DataFrame({'T1_over_45': s1, 'T1_under_45': s2, 'T2_over_45': s3, 'T2_under_45': s4, 'T3_over_45': s5, 'T3_under_45': s6, 'T4_over_45': s7, 'T4_under_45': s8})
    combined_T.plot(kind='bar')
    plt.title("Age vs T Stage")
    plt.ylabel("Count")
    st.pyplot(plt)

    A1_over_45 = df_over_45[(df_over_45['A Stage'] == 'Regional')]
    A2_over_45 = df_over_45[(df_over_45['A Stage'] == 'Distant')]
    A1_under_45 = df_under_45[(df_under_45['A Stage'] == 'Regional')]
    A2_under_45 = df_under_45[(df_under_45['A Stage'] == 'Distant')]

    s1 = A1_over_45['Status'].value_counts(normalize=True).sort_index()
    s3 = A2_over_45['Status'].value_counts(normalize=True).sort_index()
    s2 = A1_under_45['Status'].value_counts(normalize=True).sort_index()
    s4 = A2_under_45['Status'].value_counts(normalize=True).sort_index()

    combined_A = pd.DataFrame({'A1_over_45': s1, 'A1_under_45': s2, 'A2_over_45': s3, 'A2_under_45': s4})
    combined_A.plot(kind='bar')
    plt.title("Age vs A Stage")
    plt.ylabel("Count")
    st.pyplot(plt)

    sixth_1_over_45 = df_over_45[(df_over_45['6th Stage'] == 'IIA')]
    sixth_2_over_45 = df_over_45[(df_over_45['6th Stage'] == 'IIB')]
    sixth_3_over_45 = df_over_45[(df_over_45['6th Stage'] == 'IIIA')]
    sixth_4_over_45 = df_over_45[(df_over_45['6th Stage'] == 'IIIB')]
    sixth_5_over_45 = df_over_45[(df_over_45['6th Stage'] == 'IIIC')]
    sixth_1_under_45 = df_under_45[(df_under_45['6th Stage'] == 'IIA')]
    sixth_2_under_45 = df_under_45[(df_under_45['6th Stage'] == 'IIB')]
    sixth_3_under_45 = df_under_45[(df_under_45['6th Stage'] == 'IIIA')]
    sixth_4_under_45 = df_under_45[(df_under_45['6th Stage'] == 'IIIB')]
    sixth_5_under_45 = df_under_45[(df_under_45['6th Stage'] == 'IIIC')]

    s1 = sixth_1_over_45['Status'].value_counts(normalize=True).sort_index()
    s3 = sixth_2_over_45['Status'].value_counts(normalize=True).sort_index()
    s5 = sixth_3_over_45['Status'].value_counts(normalize=True).sort_index()
    s7 = sixth_4_over_45['Status'].value_counts(normalize=True).sort_index()
    s9 = sixth_5_over_45['Status'].value_counts(normalize=True).sort_index()
    s2 = sixth_1_under_45['Status'].value_counts(normalize=True).sort_index()
    s4 = sixth_2_under_45['Status'].value_counts(normalize=True).sort_index()
    s6 = sixth_3_under_45['Status'].value_counts(normalize=True).sort_index()
    s8 = sixth_4_under_45['Status'].value_counts(normalize=True).sort_index()
    s10 = sixth_5_under_45['Status'].value_counts(normalize=True).sort_index()

    plt.figure(figsize=(12, 12))
    combined_sixth = pd.DataFrame({'sixth_1_over_45': s1, 'sixth_1_under_45': s2, 'sixth_2_over_45': s3, 'sixth_2_under_45': s4, 'sixth_3_over_45': s5, 'sixth_3_under_45': s6, 'sixth_4_over_45': s7, 'sixth_4_under_45': s8, 'sixth_5_over_45': s9, 'sixth_5_under_45': s10})
    combined_sixth.plot(kind='bar')
    #plt.figure(figsize=(10, 6))
    plt.legend(fontsize=8)
    plt.title("Age vs 6th Stage")
    plt.ylabel("Count")
    st.pyplot(plt)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Flatten axes for easier looping
    axs = axes.flatten()

    # Your existing bar chart objects / Series
    plots = [combined_sixth, combined_A, combined_T, combined_N]   

    for ax, dframes in zip(axs, plots):
        dframes.plot(kind='bar', ax=ax)
        
    plt.tight_layout()
    st.pyplot(plt)




    return 0

def Question2():
    df = read_data()
    df['Tumor Size'].plot(kind='box', title="Tumor Size Outliers")
    plt.tight_layout()
    st.pyplot(plt)


    
    # Flag extreme tumor sizes (>80mm) for later analysis
    df['TumorOutlier'] = (df['Tumor Size'] > 80).astype(int)
    st.write("Number of flagged tumor outliers:", df['TumorOutlier'].sum())

    Q1 = df['Tumor Size'].quantile(0.25)
    Q3 = df['Tumor Size'].quantile(0.75)
    IQR = Q3 - Q1

    lower_extreme = Q1 - 3 * IQR
    upper_extreme = Q3 + 3 * IQR

    extreme_outliers = df[(df['Tumor Size'] < lower_extreme) | (df['Tumor Size'] > upper_extreme)]


    extreme_outlier_count = extreme_outliers.shape[0]

    st.write("Extreme outliers:", extreme_outlier_count)

    counts = extreme_outliers['Status'].value_counts().sort_index()
    nums = counts.plot(kind='bar')
    plt.bar_label(nums.containers[0], padding=3)
    plt.title("Counts of 'Status' Among Tumor Size Outlier Cases")
    plt.ylabel("Count")
    plt.xlabel("Status")
    st.pyplot(plt)

    alive_df = df[(df["Status"] == "Alive")]
    dead_df = df[(df["Status"] == "Dead")]

    plt.figure(figsize=(8,5))

    sns.kdeplot(
        data=alive_df["Tumor Size"],
        label="Alive",
        fill=True
    )

    sns.kdeplot(
        data=dead_df["Tumor Size"],
        label="Dead",
        fill=True
    )

    plt.title("KDE Plot of Tumor Size by Status")
    plt.xlabel("Tumor Size")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)

    x1 = alive_df["Tumor Size"]
    x2 = dead_df["Tumor Size"]

    # Fit KDEs manually (needed to compute intersections)
    kde1 = gaussian_kde(x1)
    kde2 = gaussian_kde(x2)

    # Shared x-grid
    xs = np.linspace(min(x1.min(), x2.min()),
                    max(x1.max(), x2.max()),
                    500)

    y1 = kde1(xs)
    y2 = kde2(xs)

    # Find intersection points
    diff = y1 - y2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    intersection_x = xs[sign_changes]

    # Plot curves
    plt.figure(figsize=(8,5))
    sns.kdeplot(x1, label='Alive')
    sns.kdeplot(x2, label='Dead')

    # Add vertical lines at intersections
    for x in intersection_x:
        plt.axvline(x, color='black', linestyle='--', alpha=0.7)

    plt.title("KDE Plot of Tumor Size by Status with Intersection Points")
    plt.legend()
    st.pyplot(plt)

    st.write("Intersection points:", intersection_x)


    

    return 0

def Question3():
    df = read_data()
    dead_df = df[(df["Status"] == "Dead")]
    dead_df_n1 = dead_df[(dead_df['N Stage'] == 'N1')]
    dead_df_n2 = dead_df[(dead_df['N Stage'] == 'N2')]
    dead_df_n3 = dead_df[(dead_df['N Stage'] == 'N3')]


    plt.figure(figsize=(8,5))

    sns.kdeplot(
        data=dead_df_n1["Survival Months"],
        label="N1",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_n2["Survival Months"],
        label="N2",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_n3["Survival Months"],
        label="N3",
        fill=True
    )

    plt.title("KDE Plot of Survival Months by N Stage")
    plt.xlabel("Survival Months")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)

    dead_df_t1 = dead_df[(dead_df['T Stage'] == 'T1')]
    dead_df_t2 = dead_df[(dead_df['T Stage'] == 'T2')]
    dead_df_t3 = dead_df[(dead_df['T Stage'] == 'T3')]
    dead_df_t4 = dead_df[(dead_df['T Stage'] == 'T4')]

    plt.figure(figsize=(8,5))

    sns.kdeplot(
        data=dead_df_t1["Survival Months"],
        label="T1",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_t2["Survival Months"],
        label="T2",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_t3["Survival Months"],
        label="T3",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_t3["Survival Months"],
        label="T4",
        fill=True
    )

    plt.title("KDE Plot of Survival Months by T Stage")
    plt.xlabel("Survival Months")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)

    dead_df_a1 = dead_df[(dead_df['A Stage'] == 'Regional')]
    dead_df_a2 = dead_df[(dead_df['A Stage'] == 'Distant')]

    plt.figure(figsize=(8,5))

    sns.kdeplot(
        data=dead_df_a1["Survival Months"],
        label="Regional",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_a2["Survival Months"],
        label="Distant",
        fill=True
    )

    plt.title("KDE Plot of Survival Months by A Stage")
    plt.xlabel("Survival Months")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)

    dead_df_sixth_1 = dead_df[(dead_df['6th Stage'] == 'IIA')]
    dead_df_sixth_2 = dead_df[(dead_df['6th Stage'] == 'IIB')]
    dead_df_sixth_3 = dead_df[(dead_df['6th Stage'] == 'IIIA')]
    dead_df_sixth_4 = dead_df[(dead_df['6th Stage'] == 'IIIB')]
    dead_df_sixth_5 = dead_df[(dead_df['6th Stage'] == 'IIIC')]

    plt.figure(figsize=(8,5))

    sns.kdeplot(
        data=dead_df_sixth_1["Survival Months"],
        label="IIA",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_sixth_2["Survival Months"],
        label="IIB",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_sixth_3["Survival Months"],
        label="IIIA",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_sixth_4["Survival Months"],
        label="IIIB",
        fill=True
    )

    sns.kdeplot(
        data=dead_df_sixth_5["Survival Months"],
        label="IIIC",
        fill=True
    )

    plt.title("KDE Plot of Survival Months by 6th Stage")
    plt.xlabel("Survival Months")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)


    return 0
    
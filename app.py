import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.set_option('deprecation.showPyplotGlobalUse', False)

model=pickle.load(open("rfc.pkl", "rb"))

def main():
    # Sidebar    
    st.sidebar.title("Find Single Employee Churn")    
    satisfaction = st.sidebar.slider("Satisfaction Level", 0, 100, 1) / 100
    last_evaluation = st.sidebar.slider("Last Evaluation", 0, 100, 1) / 100
    number_project = st.sidebar.slider("Number of Projects", 0, 20, 1)
    monthly_hours = st.sidebar.slider("Average Monthly Hours", 0,500,1)
    time_spend = st.sidebar.slider("Time Spend in Company", 0,20,1)
    work_accident = st.sidebar.selectbox("Work Accident", ["No", "Yes"])
    promotion = st.sidebar.selectbox("Promoted in Last 5 Years", ["No", "Yes"])
    departments = st.sidebar.selectbox("Department", ['IT', 'RandD', 'Accounting', 'Hr', 'Management', 'Marketing', 'Product_mng', 'Sales',                                        'Support', 'Technical'])
    salary = st.sidebar.selectbox("Salary Status", ["Low", "Medium", "High"])
    if work_accident == "Yes": work_accident = 1
    else: work_accident = 0   
    if promotion == "Yes": promotion = 1
    else: promotion = 0
    df = pd.DataFrame(data=[[satisfaction, last_evaluation, number_project, monthly_hours, time_spend, work_accident, promotion, departments, promotion]], 
                      columns=['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',                                    'Work_accident','promotion_last_5years', 'Departments', 'salary'])

    if st.sidebar.button("Analyze"):
        prediction = model.predict(df)
        if prediction == 0:
            st.sidebar.success("Stay")
        else:
            st.sidebar.error("Left")
    # Mainbar
    
    html_temp = """
                <div style="padding:1.5px">
                    <h1 style="color:black;text-align:center;">Employee Attrition Prediction App</h1>
                </div><br>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    
    
    df2 = load_data()
    if st.checkbox("Dataset Analyze"):
        select1 = st.selectbox("Please select a section:", ["", "Head", "Describe"])
     
        if select1 == "Head":
            st.table(df2.head())
        elif select1 == "Describe":
            select2 = st.selectbox("Please select value type:", ["", "Numerical", "Categorical"])
            if select2 == "Numerical":
                st.table(df2.describe())
            elif select2 == "Categorical":
                st.table(df2[["Departments", "salary"]].describe())
    if st.checkbox("Visualization"):
            df2["left_df"]=df2["left"].replace({0:"stay", 1:"Left"})
            left_names=df2["left_df"].value_counts().index
            left_val=df2["left_df"].value_counts().values
            plt.title("Visualization of Employee Left/Stay")
            plt.pie(left_val, labels=left_names, autopct="''%1.2f%%'")
            st.pyplot()
            
            plt.title("Employee Count for Number of Projects by Status")
            sns.countplot(x="number_project", hue="left_df", data=df2)
            st.pyplot()
        
            plt.title("Employee Count for Year of Experience by Status")
            sns.countplot(x="time_spend_company", hue="left_df", data=df2)
            st.pyplot()
        
            plt.title("Employee Counts in Departments by Status")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.countplot(x="Departments", hue="left_df", data=df2)
            st.pyplot(fig)
        
            plt.title("Employee Counts of Salary Scale by Status")
            sns.countplot(x="salary", hue="left_df", data=df2)
            st.pyplot()
        
            plt.title("Employee Count of Total Accident Number by Status")
            sns.countplot(x="Work_accident", hue="left_df", data=df2)
            st.pyplot()

def load_data():
    df = pd.read_csv("HR_Dataset.csv")
    return df 

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# coding: utf-8

# In[6]:


#pip install pycaret


# In[5]:


#pip install streamlit


# In[9]:


import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *

def main():
    st.title("ProjectyCaret ")

    # Upload file
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Allow user to drop columns
        st.subheader("Data Preview")
        st.write(data.head())

        columns_to_drop = st.multiselect("Select columns to drop", data.columns)
        if columns_to_drop:
            data.drop(columns=columns_to_drop, inplace=True)

        # EDA
        perform_eda = st.checkbox("Perform Exploratory Data Analysis (EDA)")
        if perform_eda:
            selected_columns = st.multiselect("Select columns for EDA", data.columns)
            if selected_columns:
                st.write(data[selected_columns].describe())

        # Handle missing values
        handle_missing = st.radio("How to handle missing values?", ('Drop rows', 'Impute'))
        if handle_missing == 'Drop rows':
            data.dropna(inplace=True)
        elif handle_missing == 'Impute':
            # Perform imputation based on user preference
            pass  # Placeholder for imputation code

        # Encode categorical data
        encode_categorical = st.radio("How to encode categorical data?", ('One Hot Encoding', 'Label Encoding'))
        if encode_categorical == 'One Hot Encoding':
            data = pd.get_dummies(data)
        elif encode_categorical == 'Label Encoding':
            # Perform label encoding based on user preference
            pass  # Placeholder for label encoding code

        # Choose X and y variables
        X = data.drop(columns=['target_variable'], axis=1)  # Assuming target variable is known
        y = data['target_variable']

        # Detect task type
        task_type = 'classification' if len(y.unique()) <= 2 else 'regression'

        # Train models
        st.subheader("Training Models")
        if task_type == 'classification':
            setup(data, target='target_variable')
            best_model = compare_models()
        elif task_type == 'regression':
            setup(data, target='target_variable')
            best_model = compare_models()

        st.subheader("Best Model")
        st.write(best_model)

if __name__ == "__main__":
    main()


# In[ ]:





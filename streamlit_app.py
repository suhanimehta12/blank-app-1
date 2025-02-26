#%%writefile app.py
import streamlit as st
import pandas as pd

st.title("📂 CSV File Reader in Streamlit")

# Upload CSV file
uploaded_file = st.file_uploader("/content/IBMEmployee_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 **Preview of the CSV File:**")
    st.dataframe(df)

    st.write("📝 **Basic File Information:**")
    st.write(df.describe(include="all"))  # Summary statistics
    st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")

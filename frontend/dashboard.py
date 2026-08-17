import streamlit as st

def show_dashboard(df):

    st.title("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Column Names")

    for col in df.columns:
        st.write("•", col)
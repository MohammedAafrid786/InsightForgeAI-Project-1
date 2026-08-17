import streamlit as st
import pandas as pd

def show_analytics(df):

    st.title("📊 Analytics Center")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Statistics")
    st.write(df.describe())
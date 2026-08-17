import streamlit as st

def load_css():
    st.markdown("""
    <style>

    .stApp{
        background:#0f172a;
        color:white;
    }

    [data-testid="stSidebar"]{
        background:#020617;
    }

    .main-title{
        text-align:center;
        font-size:50px;
        font-weight:bold;
        color:white;
    }

    .subtitle{
        text-align:center;
        color:#94a3b8;
        font-size:18px;
    }

    </style>
    """, unsafe_allow_html=True)
import streamlit as st

def show_settings():

    st.title("⚙ Settings")

    st.write("Theme Settings")

    st.toggle("Dark Mode")

    st.toggle("AI Copilot")

    st.toggle("Notifications")
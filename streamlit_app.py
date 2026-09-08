import streamlit as st
from openai import OpenAI

lab1 = st.Page("lab1.py", title= "Lab1")
lab2 = st.Page("lab2.py", title= "Lab2")
lab3 = st.Page("lab3.py", title= "Lab3")

pg = st.navigation([lab1,lab2,lab3])
st.set_page_config(page_title="Labs manager")
pg.run()

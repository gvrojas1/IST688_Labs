import streamlit as st
from openai import OpenAI

lab1 = st.Page("lab1.py", title= "Lab1")
lab2 = st.Page("lab2.py", title= "Lab2")
lab3 = st.Page("lab3.py", title= "Lab3")
lab4 = st.Page("lab4.py", title= "Lab4")
lab5 = st.Page("lab5.py", title= "Lab5")
lab6 = st.Page("Lab6.py", title= "Lab6")

pg = st.navigation([lab1,lab2,lab3,lab4,lab5,lab6])
st.set_page_config(page_title="Labs manager")
pg.run()

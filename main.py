import streamlit as st

from model_ml import (
    get_answer
)

st.text("TalkingHeads")

text = st.text_input('Enter your question for Taras Shevchenko monument:')

answer = st.button('Get answer')

if answer:
    result = get_answer(text)
    st.text(result.result)

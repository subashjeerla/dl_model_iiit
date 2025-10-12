import streamlit as st
import pandas as pd
st.title("🎈  dl prediction model")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
df = pd.read_csv("https://raw.githubusercontent.com/subashjeerla/dl_model_iiit/refs/heads/main/real_2015")
df

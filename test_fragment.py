import streamlit as st
import time

if 'count' not in st.session_state:
    st.session_state.count = 0

@st.fragment(run_every=1)
def my_fragment():
    st.session_state.count += 1
    st.write(f"Count: {st.session_state.count}")

my_fragment()

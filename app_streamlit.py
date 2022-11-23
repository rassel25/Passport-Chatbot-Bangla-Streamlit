import streamlit as st
from streamlit_chat import message
from chat import get_response

if "history" not in st.session_state:
    st.session_state.history = []

st.title("পাসপোর্ট অফিস চ্যাটবট")

def generate_answer():
    user_message = st.session_state.input_text
    message_bot = get_response(user_message)

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})


st.text_input("নিচে টাইপ করুন", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    message(**chat)  # unpacking


# use  py -m streamlit run app_streamlit.py  to run streamlit file
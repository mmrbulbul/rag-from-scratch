from threading import Thread

import streamlit as st
from transformers import TextIteratorStreamer

from rag_systems.llms.hf_api import llm_agent, tokenizer
from rag_systems.utils.utils import create_prompt

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Basic Chatbot')


streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

kwargs = dict(streamer=streamer)


def get_response(msg):
    th = Thread(target=llm_agent, args=(
        create_prompt(question=msg), ), kwargs=kwargs)
    th.start()
    for char in streamer:
        yield char


def simple_chat_ui_app():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            bot_response = st.write_stream(get_response(prompt))

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_response})

    return 0


if __name__ == "__main__":

    simple_chat_ui_app()

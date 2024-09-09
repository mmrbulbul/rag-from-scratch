import streamlit as st

from rag_systems.llms.hf_api import pipe

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Basic Chatbot')

template = [
    {"role": "system", "content": "You are a helpful AI assistant. Keep your answer short and precise. If you don't know, just say you don't know."},
    {"role": "user", "content": ""}
]


def get_response(msg):
    print("USER: ", {msg})
    template[1].update({"content": msg})
    print("PROMPT: ", template)
    output = pipe(msg)
    resp = output[0]['generated_text']
    for char in resp:
        yield char + ""


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
            with st.spinner("waiting"):
                bot_response = st.write_stream(get_response(prompt))
                # print("BOT: ", {bot_response})

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_response})

    return 0


if __name__ == "__main__":

    simple_chat_ui_app()

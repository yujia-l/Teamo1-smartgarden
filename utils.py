import os
import json
import openai
import random
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from single_round import state_dict, stage_dict

logger = get_logger('Langchain-Chatbot')

history_store = {} # for conversational history storage
status_store = {} # for conversational status storage

# set the openai api key as the environment variable
if os.path.exists("./openai.key"):
    os.environ["OPENAI_API_KEY"] = open("./openai.key").read().strip()
else:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_KEY']

#decorator
def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):
        # to clear chat history after switching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "你好！😊 我是M，你的创意问题解决项目导师,也是你们的学习伙伴!今天我们将一起开始一个以水资源为主题的项目式学习。这个项目将分为六个阶段，分别是“发现问题”、“信息搜集”、“定义问题”、“创想方案”、“方案评估“和“方案实践”. 每个阶段我们都会深入探讨水资源相关的问题，并且最终形成一个方案。"}]
        for msg in st.session_state["messages"]:
            if msg["role"] == "assistant":
                st.chat_message(msg["role"], avatar="./assets/ta_f.png").write(msg["content"])
            else:
                st.chat_message(msg["role"]).write(msg["content"])

        # to access the global variable
        if "session_id" in st.session_state:
            status = get_session_status(st.session_state["session_id"])
            if not status:
                status_store[st.session_state["session_id"]] = {
                    "stage_id": st.session_state["stage_id"],
                    "state_ids": st.session_state["state_ids"],
                    "student_type": st.session_state["student_type"],
                    "strategy_history": st.session_state["strategy_history"]
                }
            else:
                st.session_state["stage_id"] = status["stage_id"]
                st.session_state["state_ids"] = status["state_ids"]
                st.session_state["student_type"] = status["student_type"]
                st.session_state["strategy_history"] = status["strategy_history"]

        else:
            if "stage_id" not in st.session_state:
                st.session_state["stage_id"] = 0
            if "state_ids" not in st.session_state:
                st.session_state["state_ids"] = []
            if "student_type" not in st.session_state:
                st.session_state["student_type"] = 0
            if "strategy_history" not in st.session_state:
                st.session_state["strategy_history"] = []

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def get_session_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

def get_session_status(session_id: str):
    if session_id not in status_store:
        return False
    return status_store[session_id]

def write_session_status(session_id: str, stage_id: int, state_ids: list, student_type: int, strategy_history: list):
    if session_id not in status_store.keys():
        status_store[session_id] = {}
    status_store[session_id]["stage_id"] = stage_id
    status_store[session_id]["state_ids"] = state_ids
    status_store[session_id]["student_type"] = student_type
    status_store[session_id]["strategy_history"] = strategy_history

def access_global_var(func):
    def execute(*args, **kwargs):
        global history_store
        return func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author, avatar="./assets/ta_f.png").write(msg) if author == "assistant" else st.chat_message(author).write(msg)

def configure_user_session():
    # let user input a number as the session id
    session_id = st.sidebar.text_input("Session ID", key="SESSION_ID", disabled=(True if "session_id" in st.session_state else False))
    if not session_id:
        if "session_id" not in st.session_state:
            # if the user does not input a session id, generate a random one
            session_id = str(random.randint(100000,999999))
            st.session_state["session_id"] = session_id
        else:
            session_id = st.session_state["session_id"]
    else:
        st.session_state["session_id"]  = session_id

    st.sidebar.write(f"Current Session ID: {session_id}")
    return session_id

def configure_info():
    st.sidebar.markdown("### Information")
    st.sidebar.write(f"阶段: {stage_dict[st.session_state.stage_id]["name"]}")
    st.sidebar.progress(float(st.session_state.stage_id)/6, "阶段进度")
    if "state_ids" in st.session_state:
        for state_id in st.session_state["state_ids"]:
            st.sidebar.write(f"状态: {state_dict[state_id]["name"]}")
            b1, b2 = st.sidebar.columns(2)
            with b1:
                button_1 = st.button("👍", key=random.randint(100000,999999))
            with b2:
                button_2 = st.button("👎", key=random.randint(100000,999999))
            
    else:
        st.sidebar.write(f"状态: {state_dict[state_id]["name"]}")
    
def configure_download():
    messages = st.session_state["messages"]
    if len(messages) > 2:
        export = []
        for msg in messages:
            export.append({"role": msg["role"], "content": msg["content"]})
        # convert the export list to json
        export = json.dumps(export, indent=4, ensure_ascii=False)
        st.sidebar.download_button(
            label="Save Chat History",
            data=export,
            file_name=f"chat_history_{st.session_state["session_id"]}.json",
            mime="application/json"
        )

def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY"
        )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()

    model = "gpt-4o-mini"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [{"id": i.id, "created":datetime.fromtimestamp(i.created)} for i in client.models.list() if str(i.id).startswith("gpt")]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model",
            options=available_models,
            key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model, openai_api_key

def configure_llm():
    available_llms = ["gpt-4o-mini","use your openai api key"]
    llm_opt = st.sidebar.radio(
        label="LLM",
        options=available_llms,
        key="SELECTED_LLM"
        )

    if llm_opt == "gpt-4o-mini":
        llm = ChatOpenAI(model_name=llm_opt, temperature=0, streaming=True, api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        model, openai_api_key = choose_custom_openai_key()
        llm = ChatOpenAI(model_name=model, temperature=0, streaming=True, api_key=openai_api_key)
    return llm

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
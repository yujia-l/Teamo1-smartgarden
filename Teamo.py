import time
import utils
import streamlit as st
from streaming import StreamHandler

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from load_prompts import contextualize_q_prompt, get_qa_prompt
from load_database import setup_docs
from single_round import status_detection, strategy_selection
from utils import get_session_history

st.set_page_config(page_title="项目式学习助教", page_icon="🧑‍🏫")
st.header('项目式学习助教')
st.write('欢迎使用项目式学习助教！')

print("********** Starting the chatbot **********")
session_id = "999999"

class CustomChatbot:
    def __init__(self):
        utils.sync_st_session()
        self.info = utils.configure_info()
        self.llm = utils.configure_llm()
        
        # Set up the prompt template
        retriever = setup_docs()
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )
        print("Finished setting up the history-aware retriever with document loaded.")

    @utils.access_global_var
    def setup_chain(self, stage_id, urge_state_id, best_strategy_id, student_type):
        self.question_answer_chain = create_stuff_documents_chain(self.llm, get_qa_prompt(stage_id, urge_state_id, best_strategy_id, student_type))
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return self.conversational_rag_chain
    
    @utils.enable_chat_history
    def main(self):
        global global_status        
        st.session_state.last_active_time = time.time()
        chain = self.setup_chain(0, 0, 0, 0)

        user_query = st.chat_input(placeholder="欢迎提出任何问题！")
        
        if user_query:
            st.session_state.last_active_time = time.time()  # Reset the timer on new user input
            utils.display_msg(user_query, 'user')

            # preprocess the user query to detect the status
            status_detection_output = status_detection(st.session_state.messages, st.session_state.stage_id, st.session_state.state_ids)
            st.session_state.stage_id = status_detection_output.stage_id
            st.session_state.state_ids = status_detection_output.state_ids
            st.session_state.student_type = status_detection_output.student_type

            # get the strategies and make a selection
            strategy_selection_output = strategy_selection(st.session_state.messages, st.session_state.state_ids)
            urge_state_id = strategy_selection_output.urge_state_id
            best_strategy_id = strategy_selection_output.best_strategy_id

            chain = self.setup_chain(st.session_state.stage_id, urge_state_id, best_strategy_id, st.session_state.student_type)

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    input = {
                        "input": user_query,
                        },
                    config = {
                        "configurable": {"session_id": session_id}, 
                        "callbacks": [st_cb]
                        }
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})

                st.rerun()  # Rerun the app to update the chat

        # Check for inactivity
        while not user_query:
            st.session_state.inactive = self.check_for_inactivity(chain)
            # TODO: Delete if wish to check for inactivity continuously
            if st.session_state.inactive:
                break

    def check_for_inactivity(self, chain):
        current_time = time.time()
        if "last_active_time" in st.session_state and (current_time - st.session_state.last_active_time > 10):
            st.session_state.last_active_time = current_time  # Reset timer
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    {"input": "当前学生没有说话，请主动进行干预。"},
                    config={"configurable": {"session_id": session_id}, "callbacks": [st_cb]}
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
            return True
        return False

if __name__ == "__main__":
    obj = CustomChatbot()
    obj.main()
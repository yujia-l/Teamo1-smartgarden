import time
import utils
import random
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from streaming import StreamHandler

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(page_title="创意问题解决导师", page_icon="🧑‍🏫")
st.header('创意问题解决导师')

from load_prompts import contextualize_q_prompt, get_qa_prompt
from load_database import setup_docs
from structured_query import status_detection, strategy_selection, valid_strategy_ids, stage_dict
from utils import get_session_history, write_session_status, write_google_sheet


print("********** Starting the chatbot **********")

def stt_callback():
    if "stt_output" not in st.session_state:
        st.session_state.stt_output = ""
    st.write(st.session_state.stt_output)

class Teamo:
    def __init__(self):
        utils.sync_st_session()
        self.session_id = utils.configure_user_session()
        self.info = utils.configure_info()
        self.llm = utils.configure_llm()
        utils.configure_download()
        
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
    
        audio_input = speech_to_text(
            language='zh-CN',
            start_prompt="🎙️ 语音输入",
            stop_prompt="✅ 输入完毕",
            just_once=True,
            use_container_width=True,
            callback=stt_callback,
            args=(),
            kwargs={},
            key=None
        )

        # Insert the audio input into the chat input box
        js = f"""
            <script>
                function insertText(dummy_var_to_force_repeat_execution) {{
                    var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                    nativeInputValueSetter.call(chatInput, "{audio_input if audio_input else ""}");
                    var event = new Event('input', {{ bubbles: true}});
                    chatInput.dispatchEvent(event);
                }}
                insertText({len(st.session_state.messages)});
            </script>
            """
        st.components.v1.html(js)

        if user_query:
            st.session_state.last_active_time = time.time()  # Reset the timer on new user input
            utils.display_msg(user_query, 'user')
            write_google_sheet(self.session_id)

            # preprocess the user query to detect the status
            status_detection_output = status_detection(st.session_state.messages, st.session_state.stage_id, st.session_state.state_ids)
            
            # validate the status
            if status_detection_output.stage_id in range(7):
                st.session_state.stage_id = status_detection_output.stage_id 
            # validate the state ids
            valid_state_ids = [state_id for state_id in status_detection_output.state_ids if state_id in range(27)]
            st.session_state.state_ids = valid_state_ids
            # validate the student type
            if status_detection_output.student_type in range(2):
                st.session_state.student_type = status_detection_output.student_type

            # get the strategies and make a selection
            strategy_selection_output = strategy_selection(st.session_state.messages, st.session_state.state_ids)
            
            # validate the urge state id, if not in the state ids, randomly select one
            if strategy_selection_output.urge_state_id in st.session_state.state_ids:
                urge_state_id = strategy_selection_output.urge_state_id
            else:
                urge_state_id = random.choice(st.session_state.state_ids)

            # validate the best strategy id, if not in the valid strategy ids, randomly select one from the urge state
            if strategy_selection_output.best_strategy_id in valid_strategy_ids(st.session_state.state_ids):
                best_strategy_id = strategy_selection_output.best_strategy_id
            else:
                valid_strategies = valid_strategy_ids([urge_state_id])
                best_strategy_id = random.choice(valid_strategies) if valid_strategies != [] else 0

            write_session_status(self.session_id, st.session_state.stage_id, st.session_state.state_ids, st.session_state.student_type, st.session_state.strategy_history)

            chain = self.setup_chain(st.session_state.stage_id, urge_state_id, best_strategy_id, st.session_state.student_type)

            with st.chat_message("assistant", avatar="./assets/ta_f.png"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    input = {
                        "input": user_query,
                        },
                    config = {
                        "configurable": {"session_id": self.session_id}, 
                        "callbacks": [st_cb]
                        }
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
                write_google_sheet(self.session_id)
                st.rerun()

        # Check for inactivity
        while not user_query:
            st.session_state.inactive = self.check_for_inactivity(chain)
            if st.session_state.inactive:
                break

    def check_for_inactivity(self, chain):
        current_time = time.time()
        if "last_active_time" in st.session_state and (current_time - st.session_state.last_active_time > stage_dict[st.session_state.stage_id]["wait_time"]):
            st.session_state.last_active_time = current_time  # Reset timer
            st.session_state.state_ids.append(1)
            
            # get the strategies and make a selection
            strategy_selection_output = strategy_selection(st.session_state.messages, st.session_state.state_ids)
            
            # validate the urge state id, if not in the state ids, randomly select one
            if strategy_selection_output.urge_state_id in st.session_state.state_ids:
                urge_state_id = strategy_selection_output.urge_state_id
            else:
                urge_state_id = random.choice(st.session_state.state_ids)
            
            # validate the best strategy id, if not in the valid strategy ids, randomly select one from the urge state
            if strategy_selection_output.best_strategy_id in valid_strategy_ids(st.session_state.state_ids):
                best_strategy_id = strategy_selection_output.best_strategy_id
            else:
                valid_strategies = valid_strategy_ids([urge_state_id])
                best_strategy_id = random.choice(valid_strategies) if valid_strategies != [] else 0
            write_session_status(self.session_id, st.session_state.stage_id, st.session_state.state_ids, st.session_state.student_type, st.session_state.strategy_history)

            chain = self.setup_chain(st.session_state.stage_id, urge_state_id, best_strategy_id, st.session_state.student_type)

            with st.chat_message("assistant", avatar="./assets/ta_f.png"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    {"input": "当前学生没有说话"},
                    config={"configurable": {"session_id": self.session_id}, "callbacks": [st_cb]}
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
                write_google_sheet(self.session_id)
            return True
        return False

if __name__ == "__main__":
    obj = Teamo()
    obj.main()
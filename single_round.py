import os
import json
from pydantic import BaseModel
from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets['OPENAI_KEY'].strip())

def idx_string_to_int(d):
    new_d = {}
    for key, value in d.items():
        new_d[int(key)] = value
    return new_d

with open('./database/stage.json', 'r') as f:
    stage_dict = idx_string_to_int(json.load(f))
with open('./database/state.json', 'r') as f:
    state_dict = idx_string_to_int(json.load(f))
with open('./database/strategy.json', 'r') as f:
    strategy_dict = idx_string_to_int(json.load(f))

class statusDectectionOutput(BaseModel):
    stage_id: int
    state_ids: list[int]
    student_type: int

class strategySelectionOutput(BaseModel):
    urge_state_id: int
    best_strategy_id: int

def get_state_desciption():
    state_description = ''''''
    for key, value in state_dict.items():
        if key == 0 or key == 1:
            continue
        state_description += f'''状态编号{key}: {value['name']}， 定义为{value['description']}\n'''
    return state_description

def get_status_detect_prompt(stage_id, state_ids):
    return f'''
    当前处于第{stage_id}阶段-{stage_dict[stage_id]["name"]}。当前阶段的定义是{stage_dict[stage_id]["description"]}。
    下一阶段是{stage_dict[stage_id+1]["name"]}，定义为{stage_dict[stage_id+1]["description"]}。
    结合已经发生的对话信息，判断是否进入下一阶段。如果维持当前阶段，输出参数 stage_id 为 {stage_id}，否则输出参数 stage_id 为 {stage_id+1}。

    如果没有进入下一阶段，学生的状态可能为：
    {get_state_desciption()} 

    学生之前的状态列表为{state_ids}，请根据新的对话记录，更新并输出学生状态列表 state_ids。

    如果进入下一阶段，学生的状态为“开始”，更新并输出学生状态列表state_ids=[0]。

    根据历史记录中学生提供的问卷结果，判断学生类型，返回参数 student_type，如果问卷得分高于20分则输出 1，低于20分则输出 0
    '''

def get_strategy_desciption(state_ids):
    strategy_description = ''''''
    for state in state_ids:
        strategy_description += f'''状态编号{state}:{state_dict[state]['name']}，对应策略:\n'''
        for strategy in state_dict[state]['strategy_id']:
            strategy_description += f'''策略编号{strategy}: {strategy_dict[strategy]['name']}， 定义为{strategy_dict[strategy]['description']}\n'''
    return strategy_description

def get_strategy_select_prompt(state_ids):
    return f'''
    当前学生状态和对应策略如下：{get_strategy_desciption(state_ids)}
    结合已经发生的对话信息，输出最需要解决的问题编号 urge_state_id 和对应最优策略编号 best_strategy_id
    '''

def process_chat_history(chat_history):
    output = ''''''
    for chat in chat_history:
        output += f'''{chat['role']}: {chat['content']}\n'''
    return output

def status_detection(history, stage_id, state_ids) -> statusDectectionOutput:
    history = process_chat_history(history)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": get_status_detect_prompt(stage_id, state_ids)},
            {"role": "user", "content": history},
        ],
        response_format = statusDectectionOutput,
    )
    structured_output = completion.choices[0].message.parsed
    return structured_output

def strategy_selection(history, state_ids) -> strategySelectionOutput:
    history = process_chat_history(history)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": get_strategy_select_prompt(state_ids)},
            {"role": "user", "content": history},
        ],
        response_format = strategySelectionOutput,
    )
    structured_output = completion.choices[0].message.parsed
    return structured_output
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
    {StageChangePrompt[stage_id]}
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




StageChangePrompt = {}
StageChangePrompt[0] = '''
    当学生搜集完所有问题的关键信息后,询问学生是否进入下一阶段,当学生说确认则进入下一阶段.
    \n\n
    {context}
'''

StageChangePrompt[1] = '''
    当前“发现问题”阶段,在学生提交了本阶段任务时,你需要根据问题阐述要素及示例判断学生的回答是否满足要求:
    (1)提交了五到十个相关的问题阐述;
    (2)问题阐述要素需要包含场景、对象和具体的问题现状;
    (3)每个问题是由一句话阐述清楚.示例如下:
        正确示例: 某班级学生经常浪费未喝完的瓶装水，每天会有多达50瓶水被倒掉。
        错误示例:如何设计一个智慧灌溉系统，使得灌溉能够根据天气条件和草地湿度自动调节，从而减少水资源浪费和操场积水问题？
    (4)学生提交的回答不能是我曾经生成过的内容.
    (5)在阐述问题时不能出现解决方案.
    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    \n\n
    {context}
'''

StageChangePrompt[2] = '''
    当前“信息搜集”阶段,在学生提交了本阶段任务时,你需要根判断学生的回答是否满足要求:
    (1)完成了对前面提出的3-5个问题的相关信息搜索;
    (2)写一份所选问题的现状报告;
    (3)问题现状报告包括要素如下:问题背景、问题的现象描述、关键数据与事实、问题的影响范围、已采取的措施、利益相关者意见和总结与问题陈述.

    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    \n\n
    {context}
'''

StageChangePrompt[3] = '''
    当前“定义问题”阶段,在学生提交了本阶段任务时,你需要判断学生的回答是否满足要求:
    (1)最终提交了1个相关的问题阐述;
    (2)问题阐述要素需要包含场景、对象和具体的问题现状;
    (3)问题是由一句话阐述清楚.示例如下:
    假设在“发现问题”阶段，学生已经注意到学校的灌溉系统每天按时浇水，但有时浪费了大量水资源。那么在“定义问题”阶段，他们的任务可能如下：通过调查学校的灌溉系统、了解系统的工作机制，确定造成水资源浪费的原因（如定时器设计、感应器故障等）。然后将问题陈述为：“如何改善学校的灌溉系统，使其能够根据自动调整浇水时间，从而减少水资源浪费？”
    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    \n\n
    {context}
'''

StageChangePrompt[4] = '''
    当前“创想方案”阶段,在学生提交了本阶段任务时,你需要根据问题阐述要素及示例判断学生的回答是否满足要求:
    (1)提出三个可行性高、具有原创性的问题解决方案;
    (2)方案需要包含对现象、问题的剖析，再到解决方案的构思.
    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    \n\n
    {context}
'''
    
StageChangePrompt[5] = '''
    当前“方案评估”阶段，在学生提交了本阶段任务时,你需要根据方案评估报告要求及示例判断学生的回答是否满足要求,如果满足,则进入下一阶段,如果不满足,则通过引导策略中的鼓励与动机、案例引导与示例等进行引导,直至学生提交的答案满足任务要求.
    当学生在三轮对话内内就提交了答案,你则可以通过引导策略中的深入思考与反思让学生可以提出更加具有系统性思考的方案评估报告,从而迭代生成最终方案.
    当学生提交了最终方案计划时,请你确认学生是否最终提交,提交完成后给予学生一段恭喜完成本次任务挑战的话!
    \n\n
    {context}
    '''

StageChangePrompt[6] = '''
    当前“方案实践”阶段,在学生提交了本阶段任务时,你需要判断学生的回答是否满足要求:
    (1)提交了一份完整深入的实践方案报告;
    (2)报告需要包含对选择解决问题的背景介绍、解决方案的构思、具体实施计划，以及方案实施过程总可能遇到的挑战.
    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    当学生提交了最终方案计划时,请你确认学生是否最终提交,提交完成后给予学生一段恭喜完成本次任务挑战的话!
    \n\n
    {context}
    '''
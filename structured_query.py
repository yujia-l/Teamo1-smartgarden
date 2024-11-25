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
    if stage_id == 6:
        return f'''
        当前处于第{stage_id}阶段-{stage_dict[stage_id]["name"]}。当前阶段的定义是{stage_dict[stage_id]["description"]}。
        输出参数 stage_id 为 {stage_id}。

        学生的状态可能为：
        {get_state_desciption()} 

        学生之前的状态列表为{state_ids}，删除所有0和1，并请根据新的对话记录，更新并输出学生状态列表 state_ids。

        根据历史记录中学生提供的问卷结果，判断学生类型，返回参数 student_type，如果问卷得分高于20分则输出 1，低于20分则输出 0
        '''
    else:
        return f'''
        当前处于第{stage_id}阶段-{stage_dict[stage_id]["name"]}。当前阶段的定义是{stage_dict[stage_id]["description"]}。
        下一阶段是{stage_dict[stage_id+1]["name"]}，定义为{stage_dict[stage_id+1]["description"]}。
        {StageChangePrompt[stage_id]}
        结合已经发生的对话信息，判断是否进入下一阶段。如果维持当前阶段，输出参数 stage_id 为 {stage_id}，否则输出参数 stage_id 为 {stage_id+1}。

        如果没有进入下一阶段，学生的状态可能为：
        {get_state_desciption()} 

        学生之前的状态列表为{state_ids}，删除所有0和1，并请根据新的对话记录，更新并输出学生状态列表 state_ids。

        如果进入下一阶段，学生的状态为“开始”，更新并输出学生状态列表state_ids=[0]。

        根据历史记录中学生提供的问卷结果，判断学生类型，返回参数 student_type，如果问卷得分高于20分则输出 1，低于20分则输出 0
        '''

def valid_strategy_ids(state_ids):
    valid_strategies = set()
    for state in state_ids:
        for strategy in state_dict[state]['strategy_id']:
            valid_strategies.add(strategy)
    return list(valid_strategies)


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

def query_openai(prompt, history, format):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history},
        ],
        response_format = format,
    )
    output = completion.choices[0].message.parsed
    return output

def status_detection(history, stage_id, state_ids, retry=3) -> statusDectectionOutput:
    history = process_chat_history(history)
    prompt = get_status_detect_prompt(stage_id, state_ids)
    while retry > 0:
        try:
            output = query_openai(prompt, history, statusDectectionOutput)
            break
        except Exception as e:
            retry -= 1
            if retry == 0:
                raise e
    return output

def strategy_selection(history, state_ids, retry=3) -> strategySelectionOutput:
    history = process_chat_history(history)
    prompt = get_strategy_select_prompt(state_ids)
    while retry > 0:
        try:
            output = query_openai(prompt, history, strategySelectionOutput)
            break
        except Exception as e:
            retry -= 1
            if retry == 0:
                raise e
    return output


StageChangePrompt = {}
StageChangePrompt[0] = '''
    当你搜集完学生的关键信息后，进入下一阶段。
    \n\n
    {context}
'''

StageChangePrompt[1] = '''
    当前“发现问题”阶段,在学生提交了本阶段任务时,如果出现以下情况,则必须不能进入下一阶段:
    (1)没有提交三到五个相关的问题阐述;
    (2)没有包含问题阐述要素如场景、对象和具体的问题现状等;
    (3)每个问题没有由一句话阐述清楚,而是使用关键词;
    (4)学生提交的回答是我曾经生成过的内容,直接复制粘贴而成;
    (5)在阐述问题时出现了解决方案,而没有聚焦于问题现象本身.

    如果满足以下所有要求,则可以进入下一阶段.:
    (1)提交了三到五个相关的问题阐述;
    (2)问题阐述要素需要包含场景、对象和具体的问题现状;
    (3)每个问题是由一句话阐述清楚.
    (4)学生提交的回答不能是我曾经生成过的内容.
    (5)在阐述问题时不能出现解决方案.
    \n\n
    {context}
'''

StageChangePrompt[2] = '''
    当前“定义问题”阶段,在学生提交了本阶段任务时,你需要判断学生的回答是否满足要求:
    (1)最终提交了1个相关的问题阐述;
    (2)问题阐述要素需要包含场景、对象和具体的问题现状、问题主要原因、问题改进方向;
    (3)问题是由一句话阐述清楚.示例如下:
    例如在“发现问题”阶段,学生注意到家中的空调系统每天按时开启和关闭，但有时房间无人时空调仍在运行，导致电能浪费。
    那么“定义问题”阶段的任务可能如下：通过调查家庭的空调系统，了解其工作机制，确定造成电能浪费的原因（如定时器设置不合理、缺乏运动感应功能等）。然后将问题陈述为：“如何改进家庭的空调系统，使其能够根据房间内的实际活动情况自动调节运行时间，从而减少电能浪费？”

    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    \n\n
    {context}
'''

StageChangePrompt[3] = '''
    当前“创想方案”阶段,在学生提交了本阶段任务时,你需要根据问题阐述要素及示例判断学生的回答是否满足要求:
    (1)提出三个可行性高、具有原创性的问题解决方案;
    (2)方案需要包含对电子元器件、材料、工具的选用、外观草图（简略）的构思.
    (3)这个解决方案必须非常有创意并且能够利用一个装置可以呈现出方案出来.
    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    \n\n
    {context}
'''

StageChangePrompt[4] = '''
    当前“方案评估”阶段,在学生提交了本阶段任务时,你需要判断学生的回答是否满足要求:
    (1)对上一个阶段创想出来的三个方案进行评估,并生成一份评估报告和方案打分;
    (2)评估报告需要起码包含问题的意义、方案可行性、方案创新性三个维度的自我陈述和最终版方案选择的原因.
    (3)最终选定的这一个解决方案必须能够利用一个装置可以呈现出方案出来.
    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    \n\n
    {context}
'''
    
StageChangePrompt[5] = '''
    当前“方案设计”阶段,在学生提交了本阶段任务时,你需要判断学生的回答是否满足要求:
    (1)提交了一份完整深入的实践方案报告;
    (2)报告需要包含对选择解决问题的背景介绍、产品名称、设备和工具的选用、功能规划、方案设计图.
    (3)这个解决方案必须非常有创意并且能够利用一个装置可以呈现出方案出来.
    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    \n\n
    {context}
    '''

StageChangePrompt[6] = '''
    当前“技术实践”阶段,在学生提交了本阶段任务时,你需要判断学生的回答是否满足要求:
    (1)提交了一份完整深入的实践方案报告;
    (2)报告需要包含对选择解决问题的背景介绍、产品名称、设备和工具的选用、功能规划、方案设计图.
    (3)这个解决方案必须非常有创意并且能够利用一个装置可以呈现出方案出来.
    如果满足所有要求,则进入下一阶段.如果少于五轮对话，则不能进入下一阶段.
    当学生提交了最终方案计划时,请你确认学生是否最终提交,提交完成后给予学生一段恭喜完成本次任务挑战的话!
    \n\n
    {context}
    '''
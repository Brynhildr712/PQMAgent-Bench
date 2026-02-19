import time
import logging
import json
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from openai import OpenAI
import re
from tqdm import tqdm

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


# 获取模型响应
def get_response_from_model(messages, model="deepseek-v3", retries=3, backoff=2):
    """
    调用大模型生成响应，支持流式响应以兼容某些需要流式对话的API
    """
    # 确保第一个消息是 "user" 或者 "system"
    if messages[0]["role"] != "user" and messages[0]["role"] != "system":
        raise ValueError("对话的第一条消息必须是 'user' 或 'system'！")

    # 确保 "user" 和 "assistant" 交替
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            raise ValueError(f"对话格式错误，第 {i + 1} 条消息 '{messages[i]['role']}' 不能与上一条相同！")

    for attempt in range(retries):
        try:
            # 修改为流式响应
            #print(messages)
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                stream=True  # 启用流式响应
            )

            # 收集流式响应的所有部分
            response_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content

            # 正则表达式匹配 <think>...</think> 格式的内容
            cleaned_response = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL)

            return cleaned_response.strip()
        except Exception as e:
            logging.error(f"模型请求失败（尝试 {attempt + 1}/{retries}）：{str(e)}")
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2  # 指数退避
            else:
                return f"抱歉，发生了错误：{str(e)}"


class Teacher:
    def __init__(self, knowledge_base_file):
        with open(knowledge_base_file, 'r', encoding='utf-8') as file:
            self.knowledge_base = json.load(file)  # 从 JSON 文件加载知识库

    def design_dialogue(self, knowledge_item):
        # 计算年级


        sys_prompt = (
            f"【任务说明】我将提供一段知识点和一段小学课堂上师生关于这个知识点展开的对话，请分析学生从这段知识点中能掌握什么知识。\n\n"
            f"【知识点和具体内容】\n"
            f"知识点：{knowledge_item['points']}\n"
            f"知识点具体内容：{knowledge_item['content']}\n"
            f"【对话内容】\n"
            f"{knowledge_item['output']}\n"
            f"【分析要求】\n"
            f"1、对话内容是学生和老师围绕知识点展开的对话。\n"
            f"2、从这段对话中，学生可以学到一些知识，但可能对一些知识掌握并不完全。\n"
            f"3、同时，学生也有可能产生一些错误的认知。\n"
            f"4、你的任务是分析这段对话，指出学生能从这段对话中学会了什么，有没有产生错误的认知，然后分析学生还有哪些关于该知识点的内容尚未学会。\n"
            f"5、你需要回答的内容要满足以下逻辑：“学生对...已经完全掌握，对...仅片面的了解了一小部分，对...尚不清楚，对...有着错误的认知”（仅展示逻辑，并非规定格式）。\n"
            f"6、上面这段仅展示逻辑，即：同时指出已掌握的、未掌握的、错误掌握的机器掌握程度，并非规定格式，具体总结内容请你自行分析。\n"
            f"【输出格式规范】\n"
            f"分析：（首先分析一下这段对话涵盖了哪些知识点，有哪些关于该知识点的内容没有提到或提到的并不完全，以及学生有没有产生什么错误的认知或有什么没有学到的）\n"            
            f"总结：（对学生经过这场对话后的状态进行一个分析和定义。）\n"
        )

        # 获取模型生成的对话
        response = get_response_from_model([
            {"role": "system", "content": sys_prompt}
        ])

        return response


# 示例: 读取知识库并生成对话
knowledge_base_file = r"other_method_RoleLLM/RoleLLM_dialogue.json"
output_file = r"other_method_RoleLLM/RoleLLM_dialogue_real2.json"

teacher = Teacher(knowledge_base_file)

# 读取知识库
with open(knowledge_base_file, 'r', encoding='utf-8') as file:
    knowledge_base = json.load(file)

# 初始化输出数据列表
output_data = []

# 动态更新写入函数
def update_output_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 处理所有知识点并生成对话，显示进度条
print("开始从头生成所有对话...")
with tqdm(total=len(knowledge_base), desc="生成对话") as pbar:
    for item in knowledge_base:
        dialogue = teacher.design_dialogue(item)

        output_data.append({
            "id": item["id"],
            "points": item["points"],
            "content": item["content"],
            "dialogue": dialogue  # 新增的对话字段
        })

        # 动态更新写入文件
        update_output_file(output_data, output_file)

        # 更新进度条
        pbar.update(1)

print(f"所有对话生成完成，结果已保存至 {output_file}")

'''
'''
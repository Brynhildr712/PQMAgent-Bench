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

            f"【任务说明】请同时完成以下两个任务：\n"
            f"【任务1】鉴于以下想法，请去除反事实想法或矛盾想法：\n"
            f"【任务1示例1】\n"
            f"**输入：**\n"
            f"中国的首都是北京。中国的首都是上海。美国的首都是华盛顿。美国的首都是纽约。\n"
            f"**输出：**\n"
            f"中国的首都是北京。美国的首都是华盛顿。\n"
            f"【任务1示例2】\n"
            f"**输入：**\n"
            f"迈克尔喜欢踢足球。迈克尔不喜欢踢足球。詹姆斯喜欢游泳。玛丽喜欢读书。\n"
            f"**输出：**\n"
            f"詹姆斯喜欢游泳。玛丽喜欢读书。\n"
            f"【任务2】鉴于以下思路，请将具有相同实体的相似思路进行合并：\n"
            f"【任务2示例1】\n"
            f"**输入：**\n"
            f"约翰是一名演员。约翰是一名导演。约翰是一名作家。迈克是一名教师。\n"
            f"**输出：**\n"
            f"约翰是一名演员、导演和作家。迈克是一名教师。\n"
            f"【任务2示例2】\n"
            f"**输入：**\n"
            f"迈克尔喜欢踢足球。迈克尔喜欢打篮球。詹姆斯喜欢游泳。玛丽喜欢读书。\n"
            f"**输出：**\n"
            f"迈克尔喜欢踢足球和打篮球。詹姆斯喜欢游泳。玛丽喜欢读书。\n"
            f"【输出格式规范】\n"
            f"请严格按照示例格式输出，每行一句，不要添加额外内容和说明。\n"
            f"【输入内容】\n"
            f"{knowledge_item['1_content']}"
            f"{knowledge_item['2_content']}"
            f"{knowledge_item['3_content']}"
            f"{knowledge_item['4_content']}"
            f"{knowledge_item['5_content']}"
            f"{knowledge_item['6_content']}"
            f"{knowledge_item['7_content']}"
            f"{knowledge_item['8_content']}"
            f"{knowledge_item['9_content']}"
            f"{knowledge_item['10_content']}"
            f"{knowledge_item['11_content']}"
            f"请输出：\n"
        )

        # 获取模型生成的对话
        response = get_response_from_model([
            {"role": "system", "content": sys_prompt}
        ])

        return response


# 示例: 读取知识库并生成对话
knowledge_base_file = r"other_method_TIM/triple1_pure.json"
output_file = r"other_method_TIM/triple_exam.json"

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

        # 创建新字典，保留原始字段并添加dialogue字段
        new_item = item.copy()
        new_item["dialogue"] = dialogue  # 添加新的对话字段

        output_data.append(new_item)

        # 动态更新写入文件
        update_output_file(output_data, output_file)

        # 更新进度条
        pbar.update(1)

print(f"所有对话生成完成，结果已保存至 {output_file}")
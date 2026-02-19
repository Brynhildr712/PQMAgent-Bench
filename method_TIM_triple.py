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
            f"【任务说明】给定一段对话，请提取每一对问答中的关系（主体、关系、客体）及相应文本：\n"
            f"【示例1】\n"
            f"**输入：**\n"
            f"问题：你能给我推荐一些公司吗？\n"
            f"回答：我推荐谷歌。\n"
            f"**输出：**\n"
            f"三元组：（公司，推荐，谷歌）。\n"
            f"相应文本：推荐的公司是谷歌。\n"
            f"【示例2】\n"
            f"**输入：**\n"
            f"问题：中国的首都是哪个城市？\n"
            f"回答：北京。\n"
            f"**输出：**\n"
            f"三元组：（中国，首都，北京）。\n"
            f"相应文本：中国的首都是北京。\n"
            f"【任务要求】\n"
            f"1、我将提供的输入内容是一长段对话。其中包含教师和学生的许多轮对话。\n"
            f"2、每一轮对话都按照上面的示例进行一个三元组的提取和拆分。\n"
            f"3、你站在学生的视角上，代入学生的身份。即：在提取三元组的过程中，将老师说的话作为问题，学生说的话作为回答。\n"
            f"4、严格按照文本内容进行提取，即使对话内容是错误的。\n"
            f"5、比如说，如果对话中回答为“中国的首都是上海”，你依然应该提取出“（中国，首都，上海）”和“中国的首都是上海”这样的错误内容。\n"
            f"【对话内容】\n"
            f"{knowledge_item['output']}\n"
            f"【输出格式规范】（请严格按照格式输出，不要添加额外内容）\n"
            f"三元组：\n"        
            f"相应文本：\n"       
            f"三元组：\n"        
            f"相应文本：\n"       
            f"......\n"
        )

        # 获取模型生成的对话
        response = get_response_from_model([
            {"role": "system", "content": sys_prompt}
        ])

        return response


# 示例: 读取知识库并生成对话
knowledge_base_file = r"data/dialogue_all_with_id.json"
output_file = r"other_method_TIM/triple1.json"

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
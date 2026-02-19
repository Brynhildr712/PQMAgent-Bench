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
            f"【任务说明】我将给你一些知识点和学生对其掌握情况，请根据掌握情况对知识点内容进行一个改写。\n\n"
            f"【改写要求】\n"
            f"1、知识点掌握情况为百分比，代表学生对该知识点掌握程度。\n"
            f"2、学生对该知识点的掌握程度越高，这条知识点在他的认知中就更接近正确的原文。\n"
            f"3、反之，若掌握程度较低，比如说只有60%的掌握程度，那么学生的认知中这条知识点只有60%的内容是正确的，其余的40%内容是错误的。\n"
            f"4、你的任务就是对知识点的内容进行一定程度的改写，根据掌握程度将知识点内容中的一部分改成错误知识。\n"
            f"5、简而言之就是，我需要你将这条100%正确的知识点改成 X%正确+（100-X）%错误。\n"
            f"【学生知识点掌握情况】\n"
            f"知识点1：{knowledge_item['1_content']}  学生对知识点1掌握情况：{knowledge_item['1_mastery']}\n"
            f"知识点2：{knowledge_item['2_content']}  学生对知识点2掌握情况：{knowledge_item['2_mastery']}\n"
            f"知识点3：{knowledge_item['3_content']}  学生对知识点3掌握情况：{knowledge_item['3_mastery']}\n"
            f"知识点4：{knowledge_item['4_content']}  学生对知识点4掌握情况：{knowledge_item['4_mastery']}\n"
            f"知识点5：{knowledge_item['5_content']}  学生对知识点5掌握情况：{knowledge_item['5_mastery']}\n"
            f"知识点6：{knowledge_item['6_content']}  学生对知识点6掌握情况：{knowledge_item['6_mastery']}\n"
            f"知识点7：{knowledge_item['7_content']}  学生对知识点7掌握情况：{knowledge_item['7_mastery']}\n"
            f"知识点8：{knowledge_item['8_content']}  学生对知识点8掌握情况：{knowledge_item['8_mastery']}\n"
            f"知识点9：{knowledge_item['9_content']}  学生对知识点9掌握情况：{knowledge_item['9_mastery']}\n"
            f"知识点10：{knowledge_item['10_content']}  学生对知识点10掌握情况：{knowledge_item['10_mastery']}\n"
            f"知识点11：{knowledge_item['11_content']}  学生对知识点11掌握情况：{knowledge_item['11_mastery']}\n"   
            f"【输出格式规范】（输出改写后的知识点，仅输出知识点内容，不需要添加额外说明）\n"
            f"知识点1：\n"            
            f"知识点2：\n"            
            f"......\n"       
            f"知识点11：\n"
        )

        # 获取模型生成的对话
        response = get_response_from_model([
            {"role": "system", "content": sys_prompt}
        ])

        return response


# 示例: 读取知识库并生成对话
#knowledge_base_file = r"memory/exam_3.json"
#output_file = r"memory/question_after_memory3_demo.json"

knowledge_base_file = r"memory/111.json"
output_file = r"memory/111question_after_memory3_demo.json"
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
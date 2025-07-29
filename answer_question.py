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
def get_response_from_model(messages, model="deepseek-r1", retries=3, backoff=2):
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
            f"【任务说明】我将提供一套试卷和一名学生对知识点的理解，请你模拟这名学生的知识点掌握情况答题。\n\n"
            f"【试卷内容】\n"
            f"一、单选题\n"
            f"1、{knowledge_item['xzt_1']}\n"
            f"2、{knowledge_item['xzt_2']}\n"
            f"3、{knowledge_item['xzt_3']}\n"
            f"4、{knowledge_item['xzt_4']}\n"
            f"5、{knowledge_item['xzt_5']}\n"
            f"二、判断题\n"
            f"6、{knowledge_item['pdt_1']}\n"
            f"7、{knowledge_item['pdt_2']}\n"
            f"8、{knowledge_item['pdt_3']}\n"
            f"9、{knowledge_item['pdt_4']}\n"
            f"10、{knowledge_item['pdt_5']}\n"
            f"三、简答题\n"
            f"11、{knowledge_item['jdt_1']}\n"
            f"12、{knowledge_item['jdt_2']}\n"
            f"13、{knowledge_item['jdt_3']}\n"
            f"【学生对知识点的理解】\n"
            f"核心知识点原文：{knowledge_item['1_content']}  学生掌握情况：{knowledge_item['1_mastery']}\n" 
            f"其他相关的知识点：\n"
            f"知识点1：{knowledge_item['2_content']}  学生对知识点1掌握情况：{knowledge_item['2_mastery']}\n"
            f"知识点2：{knowledge_item['3_content']}  学生对知识点2掌握情况：{knowledge_item['3_mastery']}\n"
            f"知识点3：{knowledge_item['4_content']}  学生对知识点3掌握情况：{knowledge_item['4_mastery']}\n"
            f"知识点4：{knowledge_item['5_content']}  学生对知识点4掌握情况：{knowledge_item['5_mastery']}\n"
            f"知识点5：{knowledge_item['6_content']}  学生对知识点5掌握情况：{knowledge_item['6_mastery']}\n"
            f"知识点6：{knowledge_item['7_content']}  学生对知识点6掌握情况：{knowledge_item['7_mastery']}\n"
            f"知识点7：{knowledge_item['8_content']}  学生对知识点7掌握情况：{knowledge_item['8_mastery']}\n"
            f"知识点8：{knowledge_item['9_content']}  学生对知识点8掌握情况：{knowledge_item['9_mastery']}\n"
            f"知识点9：{knowledge_item['10_content']}  学生对知识点9掌握情况：{knowledge_item['10_mastery']}\n" 
            f"知识点10：{knowledge_item['11_content']}  学生对知识点10掌握情况：{knowledge_item['11_mastery']}\n" 
            f"【答题要求】\n"
            f"1、知识点掌握情况为百分比，代表学生对该知识点掌握程度。\n"
            f"2、若学生对某个知识点的掌握情况为X%，则说明他在回答有关该知识点的问题时答对的概率为X%，有（100-X）%的概率答错。\n"
            f"3、模拟学生答题的过程中，应当完全根据学生的知识点掌握情况答题。若出现掌握情况不好却正确率很高，或掌握很好却大量答错，都是不对的。\n"
            f"4、核心知识点是与试卷题目相关性最高的知识点，学生对这条知识点的掌握程度重要性、参考价值最高。\n"
            f"5、若题目涉及多个知识点，优先按核心知识点理解进行回答，其他相关知识点作为辅助判断。\n"
            f"6、单选题均为单选。\n"
            f"7、判断题的答案限定“是/否”\n"
            f"8、简答题请根据学生知识点掌握情况进行回答。\n\n"  
            f"【输出格式规范】（以下均为格式举例，与正确答案无关。请模拟该学生进行答题。请严格按照以下格式输出，不需要添加额外的内容与说明）\n"
            f"分析：（首先分析一下该学生对知识点的理解中那些认知是错误的。接着在回答这张试卷的问题时按照学生错误的认知进行答题）\n"            
            f"1、A（单选题仅输出选项，不需要输出其他说明和内容，下同）\n"            
            f"2、B\n"            
            f"3、A\n"            
            f"4、C\n"
            f"5、A\n"
            f"6、是（判断题答题限定“是/否”，不需要输出其他说明和内容，下同）\n"
            f"7、是\n"
            f"8、是\n"
            f"9、否\n"
            f"10、否\n"
            f"11、（简答题一答题）\n"
            f"12、（简答题二答题）\n"
            f"13、（简答题三答题）\n\n"
            f"【注意】答题过程应当模拟学生对知识的理解，按照学生的掌握情况进行答题。\n"
        )

        # 获取模型生成的对话
        response = get_response_from_model([
            {"role": "system", "content": sys_prompt}
        ])

        return response


# 示例: 读取知识库并生成对话
knowledge_base_file = r"memory/exam_1.json"
output_file = r"other_model/Deepseek-r1__answer_exam_after_memory.json"

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
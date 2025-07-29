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
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
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
            f"【任务说明】我将提供一些小学科学课程的知识点，你需要围绕这些知识点设计5道选择题，5道判断题，3道简答题。\n\n"
            f"【知识点】\n"
            f"核心知识点：{knowledge_item['1_content']}\n"
            f"其他知识点：\n"
            f"知识点1：{knowledge_item['2_content']}\n"
            f"知识点2：{knowledge_item['3_content']}\n"
            f"知识点3：{knowledge_item['4_content']}\n"
            f"知识点4：{knowledge_item['5_content']}\n"
            f"知识点5：{knowledge_item['6_content']}\n"
            f"知识点6：{knowledge_item['7_content']}\n"
            f"知识点7：{knowledge_item['8_content']}\n"
            f"知识点8：{knowledge_item['9_content']}\n"
            f"知识点9：{knowledge_item['10_content']}\n"
            f"知识点10：{knowledge_item['11_content']}\n"
            f"【问题设计要求】\n"
            f"1、题目设计围绕核心知识点，并结合其他知识点进行多知识点融合出题。\n"
            f"1、多知识点融合：题目设计过程中每道题目必须至少涉及2个上面给出的知识点，设计出的题目不要仅涉及一条知识点。\n"
            f"2、选择题均为单选题，请设计三个选项。\n"
            f"3、判断题的答案限定“是/否”\n"
            f"4、简答题答案为一段话\n"
            f"5、所有题目难度适中，适合小学学生做题。\n"
            f"6、题目设计可以丰富一点，不要仅仅只有一句话。\n\n"
            f"【输出格式规范】（严格遵循以下模板，禁止添加额外说明）\n"
            f"选择题一：（选择题一的内容）\n"
            f"A：（A选项内容）\n"
            f"B：（B选项内容）\n"
            f"C：（C选项内容）\n"
            f"答案：（选择题一的答案）\n\n"
            f"选择题二：（选择题二的内容）\n"
            f"A：（A选项内容）\n"
            f"B：（B选项内容）\n"
            f"C：（C选项内容）\n"
            f"答案：（选择题二的答案）\n\n"
            f"......\n\n"
            f"判断题一：（判断题一的内容）\n"
            f"答案：（是/否）\n\n"
            f"判断题二：（判断题二的内容）\n"
            f"答案：（是/否）\n\n"
            f"......\n\n"
            f"简答题一：（简答题一内容，无需提供答案）\n"
            f"简答题二：（简答题二内容，无需提供答案）\n"
            f"简答题三：（简答题三内容，无需提供答案）\n\n"
        )

        # 获取模型生成的对话
        response = get_response_from_model([
            {"role": "system", "content": sys_prompt}
        ])

        return response


# 示例: 读取知识库并生成对话
knowledge_base_file = r"multi_exam/111.json"
output_file = r"multi_exam/question111.json"

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
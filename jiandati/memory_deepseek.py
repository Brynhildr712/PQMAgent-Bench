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
            f"【任务说明】我将提供三道简答题和学生给出的答案，根据评分标准判断学生的答案时候符合学生的memory。\n\n"
            f"【题目内容】\n"
            f"简答题：\n"
            f"1、{knowledge_item['jdt_1']}\n"
            f"学生给出的答案：{knowledge_item['a_jdt_1']}\n"
            f"2、{knowledge_item['jdt_2']}\n"
            f"学生给出的答案：{knowledge_item['a_jdt_2']}\n"
            f"3、{knowledge_item['jdt_3']}\n"
            f"学生给出的答案：{knowledge_item['a_jdt_3']}\n"      
            f"【学生memory：学生对知识点的理解】\n"
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
            f"【评分要求】\n"
            f"1、memory呈现形式为学生对知识点的理解。学生知识点掌握情况为百分比，代表学生对该知识点掌握程度。\n"
            f"2、若学生对某个知识点的掌握情况为X%，则说明他在回答有关该知识点的问题时答对的概率为X%，有（100-X）%的概率答错。\n"
            f"3、你的任务是判断学生给出的这三道题的答案是否符合学生的memory。若学生对某个知识点掌握情况不好却回答的十分优秀，或掌握情况非常好却回答错误，这都是不对的。\n"
            f"4、核心知识点是与题目相关性最高的知识点，学生对这条知识点的掌握程度重要性、参考价值最高。\n"
            f"5、若题目涉及多个知识点，优先按核心知识点理解进行分析，其他相关知识点作为辅助判断。\n"
            f"【评分标准】\n"
            f"**1. 认知结论一致性**\n"
            f"定义：评价回答的核心结论（正确 / 错误、观点 / 判断）与 memory 中记录的学生对该知识点的认知（无论学生认知本身是否正确）是否一致。\n"
            f"评分标准：\n"
            f"5 分：回答结论与 memory 中学生的认知完全一致（如 memory 记 “学生认为‘地球是方的’”，回答也明确 “地球是方的”）。\n"
            f"3 分：基本一致，但存在轻微表述偏差（如 memory 记 “学生分不清乘法和加法”，回答说 “学生觉得乘法和加法差不多”）。\n"
            f"1 分：结论完全矛盾（如 memory 记 “学生知道‘1+1=2’”，回答却说 “1+1=3”）。\n\n"
            f"**2. 掌握程度贴合度**\n"
            f"定义：评价回答在 “知识点完整性、细节丰富度” 上是否与 memory 中记录的学生掌握水平（如 “完全掌握 / 部分掌握 / 模糊不清”）匹配。\n"
            f"评分标准：\n"
            f"5 分：回答的完整性、细节量与 memory 中 “部分掌握”“完全掌握”“模糊” 等描述完全匹配（如 memory 记 “学生只记得‘三角形有 3 条边’，但不知道内角和”，回答仅提到 “3 条边”，未提内角和）。\n"
            f"3 分：基本匹配，但细节量略有偏差（如 memory 记 “学生掌握部分细节”，回答细节稍多或稍少，但未偏离核心）。\n"
            f"1 分：严重不匹配（如 memory 记 “学生对该知识点完全不懂”，回答却详细列出完整知识点）。\n\n"
            f"**3. 表达风格贴合度**\n"
            f"定义：评价回答的语言风格是否符合小学生的特征（而非成人化、专业化表达）。\n"
            f"评分标准：\n"
            f"5 分：语言简单、口语化，多用短句、儿童化词汇（如 “我觉得”“好像是”），无复杂句式或专业术语（如 memory 中无专业词，回答也未出现）。\n"
            f"3 分：整体符合，但偶尔出现 1-2 处稍显成人化的表达（如 “因此”“综上所述”）。\n"
            f"1 分：语言成人化、专业化（如使用 “综上所述”“从本质而言” 等），不符合小学生表达习惯。\n"
            f"**4. 知识范围匹配度**\n"
            f"定义：评价回答是否仅基于 memory 中包含的信息，未引入 memory 外的知识（除非 memory 存在模糊点且推测合理）。\n"
            f"评分标准：\n"
            f"5 分：回答完全基于 memory 中的信息，无任何 memory 外的知识（如 memory 中未提 “地球自转”，回答也未提及）。\n"
            f"3 分：基本基于 memory，但存在 1 处轻微的额外信息（如 memory 中仅提 “月亮会变”，回答补充 “可能是被云挡住了”，而 memory 中无相关记录，但推测合理）。\n"
            f"1 分：引入大量 memory 外的知识（如 memory 中仅记录 “学生知道月亮会圆缺”，回答却详细解释 “月球绕地球公转导致圆缺”，而 memory 中无此内容）。\n\n"
            f"**5. 认知矛盾性（反向指标）**\n"
            f"定义：评价回答与 memory 中明确记录的学生认知是否存在直接矛盾（矛盾越多，扣分越多）。\n"
            f"评分标准：\n"
            f"5 分：无任何矛盾（回答与 memory 中所有相关认知完全一致）。\n"
            f"3 分：存在 1 处轻微矛盾（如 memory 记 “学生认为‘蜜蜂是益虫’”，回答说 “蜜蜂好像不是益虫”，但表述模糊）。\n"
            f"1 分：存在明显矛盾（如 memory 明确记 “学生觉得‘1 米 = 10 厘米’”，回答却说 “1 米 = 100 厘米”）。\n" 
            f"【输出格式规范】（以下分数均为格式举例，与正确分数无关。请严格按照以下格式输出，不需要添加额外的内容与说明）\n"
            f"分析：（首先对每个指标的得分情况进行一个分析。后面的输出中请严格按照格式，仅输出分数，不要有其他额外内容）\n"            
            f"指标一：5\n"            
            f"指标二：2\n"            
            f"指标三：3\n"            
            f"指标四：4\n"
            f"指标五：5\n"
        )

        # 获取模型生成的对话
        response = get_response_from_model([
            {"role": "system", "content": sys_prompt}
        ])

        return response


# 示例: 读取知识库并生成对话
knowledge_base_file = r"data_memory1.json"
output_file = r"score/111output_memory1.json"

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
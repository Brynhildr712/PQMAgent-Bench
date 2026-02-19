import json
import re
import os
from typing import Dict, List, Any, Optional


def process_dialogue(dialogue: str) -> Dict[str, str]:
    """处理对话内容，提取13道题目的答案"""
    # 定义题目答案的正则表达式模式
    # 选择题：1、A 或 1、A（说明）
    xzt_pattern = r'(\d+)[、.]\s*([A-D])'
    # 判断题：6、是 或 6、否（说明）
    pdt_pattern = r'(\d+)[、.]\s*(是|否)'
    # 简答题：11、答案内容 或 11、答案内容（说明）
    # 修改后的简答题模式，更准确地匹配简答题内容
    jdt_pattern = r'(1[1-3])[、。]\s*((?:[^（\n]|（[^）]*）)*)'

    # 提取选择题答案
    xzt_matches = re.findall(xzt_pattern, dialogue)
    xzt_answers = {f'a_xzt_{int(num)}': answer for num, answer in xzt_matches if 1 <= int(num) <= 5}

    # 提取判断题答案
    pdt_matches = re.findall(pdt_pattern, dialogue)
    pdt_answers = {f'a_pdt_{int(num) - 5}': answer for num, answer in pdt_matches if 6 <= int(num) <= 10}

    # 提取简答题答案
    jdt_matches = re.findall(jdt_pattern, dialogue)
    jdt_answers = {}
    for num, answer in jdt_matches:
        num = int(num)
        if 11 <= num <= 13:
            # 直接使用括号内的内容
            jdt_answers[f'a_jdt_{num - 10}'] = answer.strip()

    # 合并所有答案
    all_answers = {**xzt_answers, **pdt_answers, **jdt_answers}

    # 检查是否有缺失的题目，如果有则补上空字符串
    for i in range(1, 6):
        if f'a_xzt_{i}' not in all_answers:
            all_answers[f'a_xzt_{i}'] = ''
    for i in range(1, 6):
        if f'a_pdt_{i}' not in all_answers:
            all_answers[f'a_pdt_{i}'] = ''
    for i in range(1, 4):
        if f'a_jdt_{i}' not in all_answers:
            all_answers[f'a_jdt_{i}'] = ''

    return all_answers


def process_json_file(input_path: str, output_path: str) -> None:
    """处理JSON文件，拆分dialogue字段并保存结果"""
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理每个元素
        processed_data = []
        for item in data:
            dialogue = item.get('dialogue', '')
            answers = process_dialogue(dialogue)

            # 创建新元素，保留原有的id、points、content字段
            new_item = {
                'id': item.get('id'),
                'points': item.get('points'),
                'content': item.get('content')
            }
            # 添加13个答案字段
            new_item.update(answers)

            processed_data.append(new_item)

        # 保存处理后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"处理完成，结果已保存到 {output_path}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    input_file = "other_model/gemini-2.5-pro__answer_exam_after_memory.json"
    output_file = "other_model_answer/gemini-2.5-pro__split.json"
    #input_file = "other_method_RoleLLM/RoleLLM_answer.json"
    #output_file = "other_method_RoleLLM/RoleLLM_split_multi_answer_exam_after_memory.json"

    process_json_file(input_file, output_file)
import json
import re
import os


def parse_dialogue(dialogue, item_id):
    """解析dialogue字段，提取选择题、判断题和简答题，并检测匹配失败的情况"""
    result = {}
    match_failed = False
    error_reasons = []

    # 解析选择题（保留选项，去除题干中的括号内容）
    for i in range(1, 6):
        chinese_num = ['一', '二', '三', '四', '五'][i - 1]
        # 改进的正则表达式：处理括号内容，灵活匹配选项
        pattern = r"选择题{}：(.*?)(\([^)]*\))?(\n[ABC]：.*?)(\n\n|\n)答案：(\w)".format(chinese_num)
        match = re.search(pattern, dialogue, re.DOTALL)
        if match:
            # 提取题干，去除括号内容
            question_with_bracket = match.group(1).strip()
            question = re.sub(r'\([^)]*\)', '', question_with_bracket).strip()

            options = match.group(3).strip()  # 选项从第3个捕获组获取
            answer = match.group(5).strip()  # 答案从第5个捕获组获取
            full_question = f"{question}\n{options}"
            result[f"xzt_{i}"] = full_question
            result[f"xzt_a_{i}"] = answer
        else:
            match_failed = True
            error_reasons.append(f"选择题{i}匹配失败")

    # 解析判断题（保持不变）
    for i in range(1, 6):
        chinese_num = ['一', '二', '三', '四', '五'][i - 1]
        pattern = r"判断题{}：(.*?)\n答案：(.*?)(\n\n|$)".format(chinese_num)
        match = re.search(pattern, dialogue, re.DOTALL)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            result[f"pdt_{i}"] = question
            result[f"pdt_a_{i}"] = answer
        else:
            match_failed = True
            error_reasons.append(f"判断题{i}匹配失败")

    # 解析简答题（保持不变）
    for i in range(1, 4):
        chinese_num = ['一', '二', '三'][i - 1]
        next_num = ['二', '三', '四'][i - 1] if i < 3 else "$"
        pattern = r"简答题{}：(.*?)(\n\n|$|简答题{})".format(chinese_num, next_num)
        match = re.search(pattern, dialogue, re.DOTALL)
        if match:
            question = match.group(1).strip()
            result[f"jdt_{i}"] = question
        else:
            match_failed = True
            error_reasons.append(f"简答题{i}匹配失败")

    if match_failed:
        print(f"警告：ID={item_id} 的元素解析失败，原因：{', '.join(error_reasons)}")

    return result, match_failed


def process_json_file(input_file, output_file):
    """处理JSON文件，拆分dialogue字段并保存为新文件，同时检测异常元素"""
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    total_items = len(data)
    failed_items = 0

    for item in data:
        item_id = item.get('id', '无ID')
        if 'dialogue' not in item:
            print(f"警告：ID={item_id} 的元素缺少dialogue字段")
            processed_data.append(item)
            failed_items += 1
            continue

        questions, match_failed = parse_dialogue(item['dialogue'], item_id)
        new_item = {
            'id': item['id'],
            'points': item['points'],
            'content': item['content']
        }
        new_item.update(questions)
        processed_data.append(new_item)

        if match_failed:
            failed_items += 1

    # 打印处理统计信息
    print(f"处理完成：共{total_items}个元素，{failed_items}个元素解析失败")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"已保存到 {output_file}")


if __name__ == "__main__":
    input_file = "multi_exam/question.json"
    output_file = "multi_exam/question_processed.json"
    process_json_file(input_file, output_file)
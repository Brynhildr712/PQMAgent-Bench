import json
import os
import re


def process_json_files():
    # 输入文件路径
    file1_path = "memory/question_after_memory3_demo.json"
    file2_path = "memory/exam.json"
    # 输出文件路径
    output_path = "memory/question_after_memory3.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 读取第一个文件
    with open(file1_path, 'r', encoding='utf-8') as file:
        data1 = json.load(file)

    # 读取第二个文件
    with open(file2_path, 'r', encoding='utf-8') as file:
        data2 = json.load(file)

    # 确保两个文件元素数量相同
    if len(data1) != len(data2):
        raise ValueError("两个文件中的元素数量不相同，无法一一对应处理")

    # 定义需要保留的字段（按示例顺序）
    fields_to_keep = [
        'id', 'points', 'content',
        'xzt_1', 'xzt_a_1',
        'xzt_2', 'xzt_a_2',
        'xzt_3', 'xzt_a_3',
        'xzt_4', 'xzt_a_4',
        'xzt_5', 'xzt_a_5',
        'pdt_1', 'pdt_a_1',
        'pdt_2', 'pdt_a_2',
        'pdt_3', 'pdt_a_3',
        'pdt_4', 'pdt_a_4',
        'pdt_5', 'pdt_a_5',
        'jdt_1', 'jdt_2', 'jdt_3','1_content','2_content','3_content','4_content','5_content','6_content','7_content','8_content','9_content','10_content','11_content'
    ]

    # 处理每个元素
    result = []
    for i in range(len(data1)):
        item1 = data1[i]
        item2 = data2[i]

        # 创建新的元素对象
        new_item = {}

        # 保留指定字段
        for field in fields_to_keep:
            if field in item2:
                new_item[field] = item2[field]

        # 拆分dialogue字段中的知识点（保持原有正则匹配逻辑）
        dialogue = item1['dialogue']
        pattern = r'知识点\d+：'

        # 查找所有知识点开头的位置
        matches = re.finditer(pattern, dialogue)
        start_indices = [match.start() for match in matches]

        # 收集知识点
        knowledge_points = []
        if start_indices:
            # 处理第一个知识点（可能包含前面的空白）
            first_point = dialogue[:start_indices[0]].strip()
            if first_point:  # 只添加非空内容
                knowledge_points.append(first_point)

            # 处理中间的知识点
            for j in range(len(start_indices) - 1):
                point = dialogue[start_indices[j]:start_indices[j + 1]].strip()
                if point:  # 只添加非空内容
                    knowledge_points.append(point)

            # 处理最后一个知识点
            last_point = dialogue[start_indices[-1]:].strip()
            if last_point:  # 只添加非空内容
                knowledge_points.append(last_point)
        else:
            # 如果没有匹配到，保留整个dialogue作为一个知识点
            knowledge_points = [dialogue.strip()] if dialogue.strip() else []

        # 过滤空知识点
        knowledge_points = [p for p in knowledge_points if p]

        # 确保有11个知识点
        if len(knowledge_points) != 11:
            print(f"警告：第{i + 1}个元素的知识点数量不为11，实际数量：{len(knowledge_points)}")
            print(f"知识点内容：\n{json.dumps(knowledge_points, ensure_ascii=False, indent=2)}")

        # 添加11个content字段（保持原有逻辑）
        for n, point in enumerate(knowledge_points, 1):
            # 去除"知识点n："前缀（如果有的话）
            content = re.sub(rf'知识点{n}：\s*', '', point).strip()
            new_item[f"{n}_contents"] = content

        result.append(new_item)

    # 保存为新的JSON文件
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"处理完成，新文件已保存至：{output_path}")


# 执行数据处理
if __name__ == "__main__":
    process_json_files()
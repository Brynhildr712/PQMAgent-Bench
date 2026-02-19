import json
import os
from typing import Dict, List, Any


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件并返回其内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式")
        return []


def process_elements(exam_data: List[Dict[str, Any]], dialogue_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """处理exam数据，为每个元素添加对应的对话内容"""
    processed_data = []
    for element in exam_data:
        new_element = element.copy()
        for i in range(1, 12):
            id_key = f"{i}_id"
            content_key = f"{i}_content"
            if id_key in element:
                try:
                    # 注意：JSON文件中的索引是从0开始的，所以要减1
                    dialogue_index = element[id_key] - 1
                    if 0 <= dialogue_index < len(dialogue_data):
                        new_element[content_key] = dialogue_data[dialogue_index].get('dialogue', '')
                    else:
                        new_element[content_key] = f"索引 {dialogue_index + 1} 超出范围"
                except (ValueError, TypeError):
                    new_element[content_key] = "无效的ID值"
        processed_data.append(new_element)
    return processed_data


def save_processed_data(data: List[Dict[str, Any]], output_path: str) -> None:
    """将处理后的数据保存到新的JSON文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"处理后的数据已保存到 {output_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")


def main():
    """主函数：协调数据处理流程"""
    # 文件路径配置
    exam_file_path = "memory/exam_3.json"
    dialogue_file_path = "other_method_RoleLLM/RoleLLM_dialogue_pure2.json"
    output_file_path = "other_method_RoleLLM/RoleLLM_exam2.json"

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载数据
    exam_data = load_json_file(exam_file_path)
    if not exam_data:
        return

    dialogue_data = load_json_file(dialogue_file_path)
    if not dialogue_data:
        return

    # 处理数据
    processed_data = process_elements(exam_data, dialogue_data)

    # 保存处理后的数据
    save_processed_data(processed_data, output_file_path)


if __name__ == "__main__":
    main()
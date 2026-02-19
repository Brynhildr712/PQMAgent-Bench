import json
import os
from openpyxl import Workbook


def process_json_to_excel(json_file_path, excel_file_path):
    """
    处理JSON文件并将指定字段的长度之和写入Excel

    参数:
    json_file_path (str): JSON文件路径
    excel_file_path (str): 输出Excel文件路径
    """
    # 检查JSON文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件 {json_file_path} 不存在")
        return

    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        print(f"错误: 文件 {json_file_path} 不是有效的JSON格式")
        return

    # 定义要检查的字段
    fields_to_check = [
        'xzt_1', 'xzt_2', 'xzt_3', 'xzt_4', 'xzt_5',
        'pdt_1', 'pdt_2', 'pdt_3', 'pdt_4', 'pdt_5',
        'jdt_1', 'jdt_2', 'jdt_3'
    ]

    # 创建一个新的Excel工作簿
    wb = Workbook()
    ws = wb.active

    # 遍历JSON数据中的每个元素
    for row, item in enumerate(data, start=1):
        # 计算当前元素所有字段的长度之和
        total_length = 0
        for field in fields_to_check:
            # 如果字段存在，获取其长度；如果字段不存在或值为None，长度记为0
            value = item.get(field)
            total_length += len(str(value)) if value is not None else 0

        # 将长度之和写入Excel的当前行
        ws.cell(row=row, column=1, value=total_length)

    # 保存Excel文件
    try:
        wb.save(excel_file_path)
        print(f"成功将数据写入 {excel_file_path}")
    except Exception as e:
        print(f"错误: 无法保存Excel文件 - {str(e)}")


if __name__ == "__main__":
    # 定义输入和输出文件路径
    JSON_FILE_PATH = "memory/exam.json"
    EXCEL_FILE_PATH = "exam_lengths.xlsx"

    # 处理文件
    process_json_to_excel(JSON_FILE_PATH, EXCEL_FILE_PATH)
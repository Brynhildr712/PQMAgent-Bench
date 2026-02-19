import json
import os
from openpyxl import Workbook
from pathlib import Path


def calculate_probability_differences():
    # 定义输入和输出文件路径
    input_file = "other_method_MemoryBank/MB_exam_after_memory.json"
    output_dir = "excel"
    output_file = os.path.join(output_dir, "MB.xlsx")

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 计算差值列表
        differences = []
        for item in data:
            # 提取预测概率（字符串形式，如"76.03%"）
            predicted_str = item.get("1_mastery", "0%")
            # 移除百分号并转换为浮点数
            predicted = float(predicted_str.strip('%')) / 100.0

            # 提取实际概率（浮点数形式，如0.7603）
            actual = item.get("acc", 0.0)

            # 计算差值的绝对值
            diff = abs(actual - predicted)
            differences.append(diff)

        # 按升序排序
        #differences.sort()

        # 创建Excel工作簿并写入数据
        wb = Workbook()
        ws = wb.active

        # 将差值写入A列，从A1开始
        for row_idx, diff in enumerate(differences, 1):
            ws[f'A{row_idx}'] = diff

        # 保存Excel文件
        wb.save(output_file)
        print(f"成功生成Excel文件: {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误：无法解析JSON文件 {input_file}")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    calculate_probability_differences()
import json
import pandas as pd


def process_json_to_excel(json_file_path, excel_file_path):
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取所有float字段的值
        float_values = [item.get('float') for item in data if 'float' in item]

        # 创建DataFrame
        df = pd.DataFrame(float_values, columns=['float'])

        # 保存到Excel文件
        df.to_excel(excel_file_path, index=False, header=False)

        print(f"成功将{len(float_values)}个浮点数保存到{excel_file_path}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file_path} 不是有效的JSON格式")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    json_file = "result_pure_smooth/Deepseek-v3.json"
    excel_file = "result_pure_smooth/Deepseek-v3.xlsx"
    process_json_to_excel(json_file, excel_file)
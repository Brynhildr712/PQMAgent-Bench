import json


def validate_json_file(file_path):
    """验证JSON文件中的选择题和判断题答案格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 验证数据是否为列表
        if not isinstance(data, list):
            print("错误：JSON数据不是一个列表")
            return

        # 遍历每个元素进行验证
        for index, element in enumerate(data, 1):
            errors = []

            # 验证选择题答案 (a_xzt_1 到 a_xzt_5)
            for i in range(1, 6):
                field = f"a_xzt_{i}"
                if field in element:
                    value = element[field]
                    if not (isinstance(value, str) and len(value) == 1 and value in {'A', 'B', 'C'}):
                        errors.append(f"{field} 格式错误：'{value}'")

            # 验证判断题答案 (a_pdt_1 到 a_pdt_5)
            for i in range(1, 6):
                field = f"a_pdt_{i}"
                if field in element:
                    value = element[field]
                    if not (isinstance(value, str) and len(value) == 1 and value in {'是', '否'}):
                        errors.append(f"{field} 格式错误：'{value}'")

            # 如果有错误，打印元素索引和错误信息
            if errors:
                print(f"元素 {index} 有 {len(errors)} 个错误：")
                for error in errors:
                    print(f"  - {error}")

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的JSON格式")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    # 指定要验证的JSON文件路径
    file_path = "other_model_answer/gemini-2.5-pro__split.json"
    validate_json_file(file_path)
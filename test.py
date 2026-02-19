import json
import os


def filter_json_elements(source_path, target_path, target_ids):
    """
    从JSON文件中筛选指定id的元素并保存到新文件

    参数:
    source_path (str): 源JSON文件路径
    target_path (str): 目标JSON文件路径
    target_ids (list): 需要保留的id列表
    """
    try:
        # 检查源文件是否存在
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"源文件不存在: {source_path}")

        # 读取源JSON文件
        with open(source_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 筛选指定id的元素
        if isinstance(data, list):
            filtered_data = [item for item in data if item.get('id') in target_ids]
        elif isinstance(data, dict):
            # 如果JSON是一个对象，处理其值为列表的情况
            filtered_data = {}
            for key, value in data.items():
                if isinstance(value, list):
                    filtered_data[key] = [item for item in value if item.get('id') in target_ids]
                else:
                    filtered_data[key] = value
        else:
            raise ValueError("JSON数据格式不支持，应为列表或字典")

        # 创建目标文件所在目录（如果不存在）
        target_dir = os.path.dirname(target_path)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 写入筛选后的数据到新文件
        with open(target_path, 'w', encoding='utf-8') as file:
            json.dump(filtered_data, file, ensure_ascii=False, indent=2)

        print(f"成功筛选并保存 {len(filtered_data)} 个元素到 {target_path}")

    except Exception as e:
        print(f"处理JSON文件时出错: {e}")


if __name__ == "__main__":
    # 需要保留的id列表
    target_ids = [985]

    # 源文件和目标文件路径
    source_file = r"multi_exam/exam_1.json"
    target_file = r"multi_exam/111.json"

    # 执行筛选操作
    filter_json_elements(source_file, target_file, target_ids)
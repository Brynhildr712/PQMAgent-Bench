import json
import os


def calculate_averages(file_path):
    """计算JSON文件中所有元素的五个分数字段的平均值"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取JSON数据
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 检查数据是否为列表
        if not isinstance(data, list):
            raise ValueError("JSON数据不是一个列表")

        # 初始化累加器和计数器
        sums = {f"score_{i}": 0.0 for i in range(1, 6)}
        count = len(data)

        # 检查元素数量
        if count == 0:
            return {f"score_{i}": 0.0 for i in range(1, 6)}

        # 累加所有分数
        for item in data:
            for i in range(1, 6):
                key = f"score_{i}"
                # 检查字段是否存在且为数值类型
                value = item.get(key)
                if isinstance(value, (int, float)):
                    sums[key] += value
                else:
                    print(f"警告: 元素 {item} 缺少 {key} 字段或字段类型不是数值")

        # 计算平均值并保留两位小数
        averages = {key: round(total / count, 2) for key, total in sums.items()}
        return averages

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None


if __name__ == "__main__":
    file_path = r"D:\python\memory\jiandati\score\score_output_memory1.json"
    averages = calculate_averages(file_path)

    if averages:
        print("各分数字段的平均值:")
        for key, value in averages.items():
            print(f"{key}: {value}")
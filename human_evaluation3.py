import json
from collections import defaultdict


def analyze_probability_differences(file_paths):
    # 存储所有文件的统计结果
    all_results = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # 初始化计数器，0-10%, 10-20%, ..., 90-100%
            counts = defaultdict(int)
            total = 0

            # 处理每个数据项
            for item in data:
                # 提取预测概率并转换为小数
                mastery_str = item.get('1_mastery', '0%').strip('%')
                predicted = float(mastery_str) / 100.0

                # 提取实际概率
                actual = float(item.get('acc', 0))

                # 计算绝对差异百分比
                diff_percentage = abs(predicted - actual) * 100

                # 分类计数(每10%一档)
                category = min(int(diff_percentage // 10), 9)  # 0-9
                counts[category] += 1
                total += 1

            # 构建结果字典
            result = {'file': file_path}
            for i in range(10):
                lower = i * 10
                upper = lower + 10
                label = f"{lower}-{upper}%" if i < 9 else ">=90%"
                result[label] = counts[i] / total * 100 if total > 0 else 0

            all_results.append(result)

        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 不存在")
        except json.JSONDecodeError:
            print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生未知错误: {e}")

    # 输出表格
    if all_results:
        # 构建表头
        headers = ['文件名'] + [f"{i * 10}-{i * 10 + 10}%" for i in range(9)] + [">=90%"]
        # 计算总宽度
        total_width = 60 + 15 * 10

        # 打印表头
        header_format = ' '.join([f"{{:<{60 if i == 0 else 15}}}" for i in range(len(headers))])
        print(header_format.format(*headers))
        print("-" * total_width)

        # 打印数据行
        for result in all_results:
            row_data = [result['file']] + [result[f"{i * 10}-{i * 10 + 10}%"] for i in range(9)] + [result[">=90%"]]
            print(header_format.format(*[f"{d:.0f}" if isinstance(d, float) else d for d in row_data]))


# 使用示例
if __name__ == "__main__":
    # 请将下面的文件名替换为你的实际JSON文件路径
    file_names = [
        "memory/exam_after_memory1.json",
        "other_method_RoleLLM/RoleLLM_exam_after_memory.json",
        "other_method_TIM/TiM_exam_after_memory.json",
        "other_method_MemoryBank/MB_exam_after_memory.json",
        "other_method_CLLM/CLLM_exam_after_memory.json"
    ]
    analyze_probability_differences(file_names)
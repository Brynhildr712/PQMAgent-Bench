import json
import os


def process_exam_data():
    # 定义文件路径
    exam_file_path = "multi_exam/exam.json"
    memory_file_path = "memory/memory_3.json"
    output_file_path = "multi_exam/exam_3.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    try:
        # 读取exam.json文件
        with open(exam_file_path, 'r', encoding='utf-8') as exam_file:
            exam_data = json.load(exam_file)

        # 读取memory_1.json文件
        with open(memory_file_path, 'r', encoding='utf-8') as memory_file:
            memory_data = json.load(memory_file)

        # 获取memory_data的最大有效索引（从0开始）
        max_index = len(memory_data) - 1

        # 处理每个元素
        for item in exam_data:
            for i in range(1, 12):
                id_field = f"{i}_id"
                mastery_field = f"{i}_mastery"

                # 检查当前元素是否包含该id字段
                if id_field in item:
                    id_value = item[id_field]

                    # 计算实际索引（用户可能希望id值从1开始对应索引0）
                    index = id_value - 1

                    # 检查索引是否有效
                    if 0 <= index <= max_index:
                        # 获取mastery值并转换为百分比
                        mastery = memory_data[index].get("mastery", 0)
                        percentage = round(mastery * 100, 2)
                        item[mastery_field] = f"{percentage}%"
                    else:
                        # 提供更清晰的错误信息
                        if index < 0:
                            print(f"警告: {id_field}的值{id_value}过小，对应索引{index}无效")
                        else:
                            print(f"警告: {id_field}的值{id_value}对应索引{index}超出memory_data范围(0-{max_index})")
                else:
                    print(f"警告: 元素缺少{id_field}字段")

        # 保存处理后的数据
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(exam_data, output_file, ensure_ascii=False, indent=2)

        print(f"处理完成，结果已保存至{output_file_path}")

    except FileNotFoundError:
        print(f"错误: 找不到文件")
    except json.JSONDecodeError:
        print(f"错误: JSON解析失败")
    except Exception as e:
        print(f"发生错误: {e}")


# 执行数据处理
if __name__ == "__main__":
    process_exam_data()
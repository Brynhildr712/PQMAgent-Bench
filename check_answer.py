import json
import os


def check_student_answers_per_element(standard_file, student_file, result_file):
    """
    按元素单独计算学生答案正确率并生成结果文件

    Args:
        standard_file: 标准答案文件路径
        student_file: 学生答案文件路径
        result_file: 结果文件保存路径
    """
    try:
        # 读取标准答案文件
        with open(standard_file, 'r', encoding='utf-8') as f:
            standard_answers = json.load(f)

        # 读取学生答案文件
        with open(student_file, 'r', encoding='utf-8') as f:
            student_answers = json.load(f)

        # 确保两个文件长度一致
        if len(standard_answers) != len(student_answers):
            raise ValueError("标准答案和学生答案的元素数量不一致")

        result_data = []

        # 遍历每个元素进行比对
        for std_element, stu_element in zip(standard_answers, student_answers):
            correct_count = 0
            element_result = {}

            # 比对选择题答案 (xzt_a_1 ~ xzt_a_5)
            for i in range(1, 6):
                std_key = f"xzt_a_{i}"
                stu_key = f"a_xzt_{i}"
                if std_key in std_element and stu_key in stu_element:
                    if std_element[std_key] == stu_element[stu_key]:
                        correct_count += 1

            # 比对判断题答案 (pdt_a_1 ~ pdt_a_5)
            for i in range(1, 6):
                std_key = f"pdt_a_{i}"
                stu_key = f"a_pdt_{i}"
                if std_key in std_element and stu_key in stu_element:
                    if std_element[std_key] == stu_element[stu_key]:
                        correct_count += 1

            # 计算当前元素的正确率
            accuracy = round(correct_count / 10, 4)  # 保留4位小数

            # 提取标准答案中的指定字段
            standard_fields = ["id", "points", "content", "1_mastery"]
            for field in standard_fields:
                if field in std_element:
                    element_result[field] = std_element[field]

            # 提取学生答案中的指定字段
            student_fields = ["a_jdt_1", "a_jdt_2", "a_jdt_3"]
            for field in student_fields:
                if field in stu_element:
                    element_result[field] = stu_element[field]

            # 添加正确率字段
            element_result["acc"] = accuracy
            result_data.append(element_result)

        # 写入结果文件
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"答案核对完成，结果已保存至: {os.path.abspath(result_file)}")

    except FileNotFoundError:
        print(f"错误：找不到文件 - {standard_file} 或 {student_file}")
    except json.JSONDecodeError:
        print(f"错误：JSON文件解析失败，请检查文件格式")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


# 执行答案核对
if __name__ == "__main__":
    standard_file = "memory/exam_1.json"
    student_file = "other_model_answer/Qwen3-plus__split.json"
    result_file = "other_model_answer/exam_after_Qwen3-plus.json"
    #student_file = "other_method_RoleLLM/RoleLLM_split_multi_answer_exam_after_memory.json"
    #result_file = "other_method_RoleLLM/RoleLLM_exam_after_memory.json"

    check_student_answers_per_element(standard_file, student_file, result_file)
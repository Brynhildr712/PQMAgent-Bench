import json

# 提取json文件的前3个元素

# 原始JSON文件路径
original_file_path = r"memory/exam.json"
# 新的JSON文件路径
new_file_path = r"memory/exam_after_memory1_top3.json"

# 读取原始JSON文件
with open(original_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 保留前三条数据
top3_data = data[:2]

# 将前三条数据写入新的JSON文件
with open(new_file_path, 'w', encoding='utf-8') as new_file:
    json.dump(top3_data, new_file, ensure_ascii=False, indent=4)

print(f"前三条数据已保存到 {new_file_path}")

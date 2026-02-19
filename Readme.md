memory1-8为记忆状态

exam为所有题目、答案、相似的前十个、相似度

1、使用 get_memory_exam_together.py 将memory中的前十的相似度添加到exam中，得到 exam_1.json

2、使用 deepseek_make_wrong_knowledge.py 按照掌握百分比制作错误的知识点，得到 question_after_memory1_demo.json

3、使用 make_question_after_memory.py 处理demo，得到 question_after_memory1.json
（若有生成不完全的，使用 text.py 填写有缺失的id序号，重新做一个文件）

4、（提供错误知识点）使用 deepseek_answer_question.py 加载 question_after_memory1.json 回答问题，得到answer_exam_after_memory1.json
（纯提供掌握程度百分比AS）使用 deepseek_answer_question_AS.py 加载 exam.json 回答问题，得到answer_exam_after_memory1_AS.json

5、使用 split_answer.py 拆分获得的答案，两个文件都拆，得到 split_answer_exam_after_memory1.json 和 split_answer_exam_after_memory1_AS.json

6、使用 check_split_answer.py 检查一下拆分的两个文件，有没有格式不对的

7、使用 check_answer.py 对答案，得到两个结果 exam_after_memory1.json ( exam_after_memory1_AS.jsonb )

8、使用 count_brier.json 计算brier分数



###### multi表示多知识点联合出卷，没有multi的就是单一知识点出卷
###### 对于所有数据文本，文件名称中的AS、AS2表示两种消融，CLLM、MB、TIM、RoleLLM等表示不同方法对照试验

==============================
=== 文件夹列表 ===

all-MiniLM-L12-v2/
用于计算文本相似度关联度的本地小模型

cosine/
计算出来每一轮知识更新中前五十准备更新的权重

data/
一些中途半成品的数据，cosine开头为纯文本相思程度，dialogue开头为按照胡河灯方法制作的三轮（聪明、一般、弱智）对话，question开头为生成的试卷题目

excel/
不同方法下记忆点时刻所有知识的应掌握、实际掌握情况差值，范围0-1

jiandati/
计算不同方法、不同记忆点下简答题回答情况的各类文件

memory/
不同记忆点下答题的情况，存放的是各种回答的答案

multi_exam/
多知识点联合出卷

other_method_CLLM/
other_method_MemoryBank/
other_method_RoleLLM/
other_method_TIM/
other_model/
other_model_answer/
对比其他方法的回答情况和情况打分计算

result_pure/
result_pure_smooth/
不同基座模型下回答情况和情况打分计算

==============================
=== 文件列表 ===


ALL_score.py
ALL_score_top200.py
算分

Graph.py
Graph2.py
作图

Readme.md
add_memory.py
check_answer.py
check_split_answer.py
config.py
config_r1.py
count_ECE.py
count_brier.py
count_variance.py
算分过程，见上面的算法流程

cut_dialogue.py
cut_json.py
处理文件

deepseek_answer_question.py
deepseek_answer_question_AS1.py
deepseek_make_exam.py
deepseek_make_wrong_knowledge.py
答题、制作试卷的prompt

distribution_exam_after_memory1.png
distribution_exam_after_memory1_AS.png
distribution_exam_after_memory1_AS2.png
图

exam_lengths.xlsx
exam_lengths_multi.xlsx
exam_to_excel.py
试卷文字量计算

get_memory_exam_together.py
human_evaluation1.py
human_evaluation2.py
human_evaluation3.py
人工评价计算

make_cosine_ready.py
make_question_after_memory.py

method_CLLM_deepseek.py
method_RoleLLM_deepseek.py
method_RoleLLM_get_knowledge.py
method_TIM_deepseek.py
method_TIM_get_picture1.py
method_TIM_triple.py
method_memorybank_deepseek.py
method_memorybank_deepseek2.py
method_memorybank_get_picture.py
不同方法答题

myplot1.png
myplot2.png
图

othermodel_answer_question.py
picture_1.py
result_pure.py
result_to_excel.py
similarity.py
similarity_whole_dialogue.py
计算结果

split_answer.py
split_question.py
test.py
test111.py
test222.py
test_script(1).py
简单的文件样式处理
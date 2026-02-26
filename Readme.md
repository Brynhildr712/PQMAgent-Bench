memory1-8 are memory states

exam includes all questions, answers, top ten similar items, and similarity scores

1. Use get_memory_exam_together.py to add the top ten similarity scores from memory to exam, obtaining exam_1.json
2. Use deepseek_make_wrong_knowledge.py to create wrong knowledge points based on mastery percentages, obtaining question_after_memory1_demo.json
3. Use make_question_after_memory.py to process the demo, obtaining question_after_memory1.json
   (If generation is incomplete, use text.py to fill in the missing ID numbers and recreate the file)
4. (With wrong knowledge points provided) Use deepseek_answer_question.py to load question_after_memory1.json and answer questions, obtaining answer_exam_after_memory1.json
   (With only mastery percentage AS provided) Use deepseek_answer_question_AS.py to load exam.json and answer questions, obtaining answer_exam_after_memory1_AS.json
5. Use split_answer.py to split the obtained answers for both files, obtaining split_answer_exam_after_memory1.json and split_answer_exam_after_memory1_AS.json
6. Use check_split_answer.py to check the two split files for any formatting errors
7. Use check_answer.py to evaluate the answers, obtaining two results: exam_after_memory1.json (exam_after_memory1_AS.jsonb)
8. Use count_brier.json to calculate the Brier score

"multi" indicates joint exam generation with multiple knowledge points; without "multi" means single knowledge point exam generation

For all data texts, AS and AS2 in file names denote two types of ablation; CLLM, MB, TIM, RoleLLM, etc., denote controlled experiments with different methods

==============================
=== Folder List ===

all-MiniLM-L12-v2/
Local small model used to calculate text similarity correlation

cosine/
Contains the top fifty weights prepared for updating in each round of knowledge update

data/
Some intermediate semi-finished data; those starting with "cosine" are pure text similarity; those starting with "dialogue" are three-round dialogues (smart, average, slow-witted) created using Hu Heden's method; those starting with "question" are generated exam questions

excel/
Differences (range 0-1) between expected and actual mastery of all knowledge points at memory moments under different methods

jiandati/
Various files for evaluating short answer question responses under different methods and memory points

memory/
Answer records under different memory points, storing various answer files

multi_exam/
Joint exam generation with multiple knowledge points

other_method_CLLM/
other_method_MemoryBank/
other_method_RoleLLM/
other_method_TIM/
other_model/
other_model_answer/
Answer records and scoring calculations for comparison experiments with other methods

result_pure/
result_pure_smooth/
Answer records and scoring calculations under different base models

==============================
=== File List ===

ALL_score.py
ALL_score_top200.py
Score calculation

Graph.py
Graph2.py
Plotting

Readme.md
add_memory.py
check_answer.py
check_split_answer.py
config.py
config_r1.py
count_ECE.py
count_brier.py
count_variance.py
Scoring process, see algorithm flow above

cut_dialogue.py
cut_json.py
File processing

deepseek_answer_question.py
deepseek_answer_question_AS1.py
deepseek_make_exam.py
deepseek_make_wrong_knowledge.py
Prompts for answering questions and generating exams

distribution_exam_after_memory1.png
distribution_exam_after_memory1_AS.png
distribution_exam_after_memory1_AS2.png
Images

exam_lengths.xlsx
exam_lengths_multi.xlsx
exam_to_excel.py
Exam text length calculation

get_memory_exam_together.py
human_evaluation1.py
human_evaluation2.py
human_evaluation3.py
Human evaluation calculation

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
Answering under different methods

myplot1.png
myplot2.png
Images

othermodel_answer_question.py
picture_1.py
result_pure.py
result_to_excel.py
similarity.py
similarity_whole_dialogue.py
Result calculation

split_answer.py
split_question.py
test.py
test111.py
test222.py
test_script(1).py
Simple file formatting handling

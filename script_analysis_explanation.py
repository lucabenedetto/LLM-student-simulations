import ast
import pandas as pd
import matplotlib.pyplot as plt

def extract_explanation(text):
    d = ast.literal_eval(text)
    return d.get("answer explanation", "")

model = 'gpt3_5'
# model = 'gpt3_5_1106'
# model = 'gpt_4_1106'

# xl = pd.ExcelFile(data_path)
df_dict = dict()
for level in range(1, 6):
    data_path = f"./data/output/test/{model}_responses_cupa/{model}_grade_answers_prompt40_0shot_a_{level}.csv"
    # data_path = f"./data/output/test/gpt3_5_1106_responses_cupa/gpt3_5_1106_grade_answers_prompt40_0shot_a_{level}.csv"
    # data_path = f"./data/output/test/gpt4_1106_responses_cupa/gpt4_1106_grade_answers_prompt40_0shot_a_{level}.csv"
    temp_df = pd.read_csv(data_path)
# for sheet_name in xl.sheet_names[1:]:
    # temp_df = pd.read_excel(data_path, sheet_name=sheet_name)
    temp_df["explanation"] = temp_df["answer"].apply(extract_explanation)
    # temp_df_with_question = pd.merge(temp_df, df, on="q_id")
    temp_df = pd.merge(temp_df, pd.read_csv('./data/input/cupa.csv'), on='q_id')
    df_dict[level] = temp_df
    # print(temp_df[:5])

keywords = [
    "might not", "might struggle", "mistaken", "abstract", "inference", "confused", "look for", "directly", #level 1
    "follow these steps", "critical thinking", "misconception", "identify", "main idea", "able to understand", #level 2-3
    "would likely not be confused", #level 4,
    "would summarize", "trap", "pitfall", "would understand", "avoid", "would avoid"
]

report_df = pd.DataFrame(
    columns=["keyword", "student level", "tot_count"]  # , "difficulty_0", "difficulty_1", "difficulty_2"]
)
tot_counts = []
student_levels = []
keywords_col = []
# difficulty_0 = []  # TODO anto used these on RACE to consider the three difficulty levels separately.
# difficulty_1 = []
# difficulty_2 = []
for keyword in keywords:
    # print(f"**{keyword.upper()}**")
    for student_level in df_dict:
        student_levels.append(student_level)
        keywords_col.append(keyword)
        count = df_dict[student_level].loc[df_dict[student_level]['explanation'].str.contains(keyword)]
        if keyword in {'misconception'}:  # , 'might not'
            print(student_level)
            for explanation in count['explanation'].values:
                print(explanation)
            print("- - - - - - - - - -")
        # print(count)
        count_by_diff = count.groupby('target_level').agg(count=("answer", "count")).reset_index()
        tot_count = count_by_diff["count"].sum()
        if count.empty:
            count_by_diff = 0
            tot_count = 0
            # difficulty_0.append(0)
            # difficulty_1.append(0)
            # difficulty_2.append(0)
        # else:
        #     difficulty_0.append(count_by_diff.loc[count_by_diff.target_level == 0]["count"])
        #     difficulty_1.append(count_by_diff.loc[count_by_diff.target_level == 1]["count"])
        #     difficulty_2.append(count_by_diff.loc[count_by_diff.target_level == 2]["count"])

        tot_counts.append(tot_count)
        # print(f"[{keyword}] {student_level}: \n{count_by_diff}")
    # print()

report_df["keyword"] = keywords_col
report_df["student level"] = student_levels
report_df["tot_count"] = tot_counts
# report_df["difficulty_0"] = difficulty_0
# report_df["difficulty_1"] = difficulty_1
# report_df["difficulty_2"] = difficulty_2

# print(report_df)

for keyword in report_df['keyword'].unique():
    tmp_df = report_df[report_df['keyword'] == keyword]
    # print(tmp_df)
    if tmp_df['tot_count'].sum() == 0:
        continue
    fig, ax = plt.subplots()
    ax.plot(tmp_df['student level'], tmp_df['tot_count'], '*-')
    ax.set_title(f'{model} | {keyword} | total count')
    ax.set_ylabel("Total count")
    ax.set_xlabel("Student level")
    plt.savefig(f'./output_figures_analysis_explanation/{model}_{keyword}_total_count.png')
    plt.close(fig)

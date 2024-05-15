import matplotlib.pyplot as plt
import pandas as pd
import ast
import scipy
from sklearn.metrics import r2_score, mean_absolute_error


def extract_difficulty(text):
    try:
        d = ast.literal_eval(text)
    except ValueError:
        return ""
    return d.get("question level", "")


def extract_selected_answer(text):
    try:
        d = ast.literal_eval(text)
    except ValueError:
        return ""
    return d.get("index", "")


def linear_scaling(x, original_min, original_max, target_min, target_max):
    return (x - original_min) / (original_max - original_min) * (target_max - target_min) + target_min


DICT_DATASET_MIN_DIFFICULTY = {'race_pp': 0, 'arc': 3, 'cupa': 30}
DICT_DATASET_MAX_DIFFICULTY = {'race_pp': 2, 'arc': 9, 'cupa': 110}


def main(model_name, dataset_name):
    print("---------- DATASET:", dataset_name, "| MODEL:", model_name, "------------")
    df_dict = dict()
    for level in range(1, 6):
        prompt_id = 40 if dataset_name in {'cupa', 'race_pp'} else 48
        data_path = f"./data/output/test/{model_name}_responses_{dataset_name}/{model_name}_grade_answers_prompt{prompt_id}_0shot_a_{level}.csv"
        temp_df = pd.read_csv(data_path)
        temp_df["llm_difficulty"] = temp_df["answer"].apply(extract_difficulty)
        temp_df['selected_answer'] = temp_df['answer'].apply(extract_selected_answer)
        if dataset_name == 'cupa':
            temp_df = pd.merge(temp_df, pd.read_csv('./data/input/cupa.csv'), on='q_id')
        elif dataset_name == 'arc':
            temp_df = pd.merge(temp_df, pd.read_csv('./data/input/arc_test.csv'), on='q_id')
        elif dataset_name == 'race_pp':
            temp_df = pd.merge(temp_df, pd.read_csv('./data/input/race_pp_test.csv'), on='q_id')
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        temp_df['selected_answer_is_correct'] = temp_df.apply(
            lambda r: r['selected_answer'] != '' and int(r['correct_answer']) == int(r['selected_answer']), axis=1)
        temp_df = temp_df.drop(columns=['answer', 'options'])
        len_0 = len(temp_df)
        temp_df = temp_df[temp_df['llm_difficulty'].isin(['1', '2', '3', '4', '5'])]
        print(f"Sim. level {level}, num. of llm_difficulty values removed due to bad formatting = {len_0-len(temp_df)}")
        temp_df['scaled_llm_difficulty'] = temp_df.apply(
            lambda r:
            linear_scaling(int(r['llm_difficulty']), 1, 5, DICT_DATASET_MIN_DIFFICULTY[dataset_name], DICT_DATASET_MAX_DIFFICULTY[dataset_name])
            , axis=1
        )
        df_dict[level] = temp_df

    # print(df_dict[1]['llm_difficulty'])
    for simulated_level in range(1, 6):
        print("DOING SIMULATED_LEVEL", simulated_level)
        target_difficulty = [float(diff) for diff in df_dict[simulated_level]['difficulty'].values]
        llm_difficulty = [float(diff) for diff in df_dict[simulated_level]['llm_difficulty'].values]
        scaled_llm_difficulty = [float(diff) for diff in df_dict[simulated_level]['scaled_llm_difficulty'].values]
        print("Correlation (original):", scipy.stats.linregress(llm_difficulty, target_difficulty))
        # this is just to check that the rescaling didn't break anything.
        print("Correlation (scaled)  :",  scipy.stats.linregress(scaled_llm_difficulty, target_difficulty))
        print("R2:", r2_score(target_difficulty, scaled_llm_difficulty))
        print("MAE:", mean_absolute_error(target_difficulty, scaled_llm_difficulty))
        print("MAPE:", 100*mean_absolute_error(target_difficulty, scaled_llm_difficulty)/(DICT_DATASET_MAX_DIFFICULTY[dataset_name] - DICT_DATASET_MIN_DIFFICULTY[dataset_name]))
        # plt.hist(llm_difficulty)
        # plt.show()


if __name__ == "__main__":
    # model = 'gpt_4_1106'  # gpt3_5, gpt3_5_1106, gpt_4_1106
    # dataset_name = 'arc'  # cupa, race_pp, arc
    for model in ['gpt3_5', 'gpt3_5_1106', 'gpt_4_1106']:
        for dataset_name in ['cupa', 'race_pp', 'arc']:
            main(model, dataset_name)

import os

import pandas as pd

from utils import (
    get_average_accuracy_per_model,
    get_response_correctness_per_model,
    get_difficulty_dict_from_df,
    get_original_dataset,
    get_questions_answered_by_all_roleplayed_levels,
    get_student_levels_from_prompt_idx,
)
from utils_llama import get_index_from_raw_answer
from utils_plotting import (
    plot_accuracy_per_model,
    plot_accuracy_per_difficulty_per_model,
    plot_accuracy_per_difficulty_for_different_role_played_levels,
    plot_correlation_between_difficulty_and_qa_correctness,
)
from constants import (
    DIFFICULTY_LEVELS,
    RACE,
    ARC,
    CUPA,
    OUTPUT_DATA_DIR,
    GPT_3_5,
    LLAMA2_7B_CHAT,
    LLAMA2_13B_CHAT,
    VICUNA_13B_V1_5,
)

DATASET = RACE
PROMPT_IDX = 40
MODEL = VICUNA_13B_V1_5


def main():
    data_path = os.path.join(OUTPUT_DATA_DIR, f'{MODEL}_responses_{DATASET}')
    out_fig_path = os.path.join('output_figures', f'{MODEL}_{DATASET}')

    student_levels = get_student_levels_from_prompt_idx(PROMPT_IDX)
    # the 1+idx is needed for backward compatibility with files written with a previous script.
    # also, this is so noise due to some inconsistency in the output filenames, I will have to sort this out...
    if MODEL in {GPT_3_5}:
        filenames = [f"gpt3_5_grade_answers_prompt{PROMPT_IDX}_0shot_a_{1+idx}.csv" for idx, _ in enumerate(student_levels)]
    elif MODEL in {LLAMA2_7B_CHAT, LLAMA2_13B_CHAT}:
        filenames = [f"llama2_answers_prompt{PROMPT_IDX}_0shot_{1 + idx}.csv" for idx, _ in enumerate(student_levels)]
    elif MODEL in {VICUNA_13B_V1_5}:
        filenames = [f"{MODEL}_answers_prompt{PROMPT_IDX}_0shot_{1 + idx}.csv" for idx, _ in enumerate(student_levels)]
    else:
        raise ValueError()
    list_dfs = [pd.read_csv(os.path.join(data_path, filename)) for filename in filenames]
    if MODEL in {LLAMA2_7B_CHAT, LLAMA2_13B_CHAT}:
        for idx in range(len(list_dfs)):
            list_dfs[idx]['answer_index'] = list_dfs[idx].apply(lambda r: get_index_from_raw_answer(r['raw_answer']), axis=1)
            print(list_dfs[idx].value_counts('answer_index'))

    difficulty_levels = DIFFICULTY_LEVELS[DATASET]
    complete_df = get_original_dataset(DATASET)

    # to keep only the questions that are answered by all models
    set_q_ids = get_questions_answered_by_all_roleplayed_levels(list_dfs, complete_df)

    # dict that maps from qid to "true" difficulty
    difficulty_dict = get_difficulty_dict_from_df(complete_df)

    # code below compute the correctness for the different models, which is the info that I can then use to plot the eval metrics.
    avg_accuracy_per_model, avg_accuracy_per_grade_per_model = get_average_accuracy_per_model(list_dfs, set_q_ids, complete_df, difficulty_levels)
    correctness_per_model, answers_per_model = get_response_correctness_per_model(list_dfs, set_q_ids, complete_df)

    # accuracy per model
    plot_accuracy_per_model(
        avg_accuracy_per_model, student_levels, DATASET, PROMPT_IDX,
        output_filepath=os.path.join(out_fig_path, f'{PROMPT_IDX}_accuracy_per_roleplayed_level'),
    )

    if DATASET != CUPA:
        # accuracy per model per (true) difficulty level (i.e. grade)
        plot_accuracy_per_difficulty_per_model(
            avg_accuracy_per_grade_per_model, DATASET, PROMPT_IDX,
            output_filepath=os.path.join(out_fig_path, f'{PROMPT_IDX}_accuracy_per_roleplayed_level_on_different_difficulty_levels'),
        )

    # accuracy per different grades when role-playing different levels
    plot_accuracy_per_difficulty_for_different_role_played_levels(
        avg_accuracy_per_grade_per_model, student_levels, DATASET, PROMPT_IDX,
        output_filepath=os.path.join(out_fig_path, f'{PROMPT_IDX}_accuracy_per_difficulty_with_different_roleplayed_levels'),
    )

    # correlation between grade (difficulty) and QA correctness
    plot_correlation_between_difficulty_and_qa_correctness(
        correctness_per_model, difficulty_dict, DATASET, PROMPT_IDX,
        output_filepath_hexbin=os.path.join(out_fig_path, f'{PROMPT_IDX}_correlation_difficulty_and_qa_correctness_hexbin'),
        output_filepath_kdeplot=os.path.join(out_fig_path, f'{PROMPT_IDX}_correlation_difficulty_and_qa_correctness_kdeplot'),
    )


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os

from utils import (
    get_average_accuracy_per_model,
    get_response_correctness_per_model,
    get_original_dataset,
    get_questions_answered_by_all_roleplayed_levels,
    get_student_levels_from_prompt_idx,
)
from utils_plotting import (
    plot_accuracy_per_model,
    plot_accuracy_per_difficulty_per_model,
    plot_accuracy_per_difficulty_for_different_role_played_levels,
)
from constants import (
    RACE,
    CUPA,
    ARC,
    OUTPUT_DATA_DIR,
    GPT_3_5,
    GPT_3_5_1106,
    GPT_4_1106,
    TEST,
    DEV,
)

DATASET = RACE
PROMPT_IDX = 40
MODEL = GPT_3_5
SPLIT = TEST


def main():
    data_path = os.path.join(OUTPUT_DATA_DIR, SPLIT, f'{MODEL}_responses_{DATASET}')
    out_fig_path = os.path.join('../output_figures', SPLIT, f'{MODEL}_{DATASET}')

    student_levels = get_student_levels_from_prompt_idx(PROMPT_IDX)
    # the 1+idx is needed for backward compatibility with files written with a previous script.
    if MODEL in {GPT_3_5, GPT_3_5_1106, GPT_4_1106}:
        filenames = [f"{MODEL}_grade_answers_prompt{PROMPT_IDX}_0shot_a_{1+idx}.csv" for idx, _ in enumerate(student_levels)]
    else:
        raise ValueError()
    list_dfs = [pd.read_csv(os.path.join(data_path, filename)) for filename in filenames]

    complete_df = get_original_dataset(DATASET)

    # to keep only the questions that are answered by all models
    set_q_ids = get_questions_answered_by_all_roleplayed_levels(list_dfs, complete_df)

    # code below compute the correctness for the different models, which is the info that I can then use to plot the eval metrics.
    avg_accuracy_per_model, avg_accuracy_per_grade_per_model = get_average_accuracy_per_model(list_dfs, set_q_ids, complete_df)
    correctness_per_model, answers_per_model = get_response_correctness_per_model(list_dfs, set_q_ids, complete_df)

    # Analysis of the prompts that do not simulate student levels
    if PROMPT_IDX in {61, 62, 63, 64, 65, 66, 67}:
        print(f"{DATASET} | Prompt {PROMPT_IDX} --> MCQA accuracy: {np.mean(list(correctness_per_model.values()))}")

    else:
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


if __name__ == "__main__":
    main()

import os

from utils import (
    get_average_accuracy_per_model,
    get_response_correctness_per_model,
    get_difficulty_dict_from_df,
    get_original_dataset,
    get_questions_answered_by_all_roleplayed_levels,
    get_student_levels_from_prompt_idx,
)
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
    OUTPUT_DATA_DIR,
)


DATASET = RACE
PROMPT_IDX = 40
data_path = os.path.join(OUTPUT_DATA_DIR, f'gpt_responses_{DATASET}')

student_levels = get_student_levels_from_prompt_idx(PROMPT_IDX)
filenames = [f"gpt3_5_grade_answers_prompt{PROMPT_IDX}_0shot_a_{student_level}.csv" for student_level in student_levels]
filepaths = [os.path.join(data_path, filename) for filename in filenames]

difficulty_levels = DIFFICULTY_LEVELS[DATASET]
complete_df = get_original_dataset(DATASET)

# to keep only the questions that are answered by all models
set_q_ids = get_questions_answered_by_all_roleplayed_levels(filepaths, complete_df)

# dict that maps from qid to "true" difficulty
difficulty_dict = get_difficulty_dict_from_df(complete_df)

# code below compute the correctness for the different models, which is the info that I can then use to plot the eval metrics.
avg_accuracy_per_model, avg_accuracy_per_grade_per_model = get_average_accuracy_per_model(filepaths, set_q_ids, complete_df, difficulty_levels)
correctness_per_model, answers_per_model = get_response_correctness_per_model(filepaths, set_q_ids, complete_df)

# accuracy per model
plot_accuracy_per_model(avg_accuracy_per_model, student_levels, DATASET, PROMPT_IDX)

# accuracy per model per (true) difficulty level (i.e. grade)
plot_accuracy_per_difficulty_per_model(avg_accuracy_per_grade_per_model, DATASET, PROMPT_IDX)

# accuracy per different grades when role-playing different levels
plot_accuracy_per_difficulty_for_different_role_played_levels(avg_accuracy_per_grade_per_model, student_levels, DATASET, PROMPT_IDX)

# correlation between grade (difficulty) and QA correctness
plot_correlation_between_difficulty_and_qa_correctness(correctness_per_model, difficulty_dict, DATASET, PROMPT_IDX)
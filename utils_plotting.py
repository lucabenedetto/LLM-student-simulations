import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn
from typing import Optional, Tuple

from utils import (
    get_student_levels_from_prompt_idx,
    get_questions_answered_by_all_roleplayed_levels,
    get_average_accuracy_per_model,
    get_response_correctness_per_model,
)
from constants import OUTPUT_DATA_DIR

COLORS = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:olive',
    'tab:purple', 'tab:pink', 'tab:cyan', 'tab:gray', 'tab:brown',
]


def plot_accuracy_per_model(
        average_accuracy_per_model,
        role_played_levels,
        dataset_name,
        prompt_idx,
        output_filepath: str = None,
        figsize: Tuple[int, int] = (7, 5),
        output_file_extension: str = '.png',
):
    n_role_played_levels = len(role_played_levels)

    fig, ax = plt.subplots(figsize=figsize)
    # ax.bar(range(n_role_played_levels), average_accuracy_per_model)
    # ax.grid(alpha=0.5, axis='y')
    ax.plot(range(n_role_played_levels), average_accuracy_per_model)
    ax.grid(alpha=0.5, axis='both')
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_ylabel('QA accuracy')
    ax.set_xlabel('Role-played level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(role_played_levels)
    ax.set_title(f'QA accuracy per role-played level | {dataset_name} | prompt {prompt_idx}')
    if output_filepath is None:
        plt.show()
    else:
        plt.savefig(output_filepath + output_file_extension)
    plt.close(fig)


def plot_accuracy_per_difficulty_per_model(
        avg_accuracy_per_grade_per_model,
        dataset_name,
        prompt_idx,
        output_filepath: str = None,
        figsize: Optional[Tuple[int, int]] = None,
        output_file_extension: str = '.png',
):
    difficulty_levels = list(avg_accuracy_per_grade_per_model.keys())
    n_role_played_levels = len(avg_accuracy_per_grade_per_model[difficulty_levels[0]])
    for difficulty in difficulty_levels[1:]:
        if len(avg_accuracy_per_grade_per_model[difficulty]) != n_role_played_levels:
            print("WARNING!")  # TODO add raise error instead of this
    if figsize is None:
        figsize = (len(difficulty_levels)*2.5, 5)
    fig, ax = plt.subplots(1, len(difficulty_levels), figsize=figsize, sharey='all')
    for idx, grade in enumerate(difficulty_levels):
        ax[idx].set_ylim(0, 1.0)
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].grid(alpha=0.5, axis='y')
        ax[idx].bar(range(n_role_played_levels), avg_accuracy_per_grade_per_model[grade], color=COLORS[idx])
        ax[idx].plot([0, n_role_played_levels-1], [np.mean(avg_accuracy_per_grade_per_model[grade])] * 2, c='k')
        ax[idx].set_title(f'{dataset_name} | prompt {prompt_idx}\n Diff. level {grade} ')
        ax[idx].set_xlabel('Role-played level')
        ax[idx].set_ylabel('QA accuracy')
    if output_filepath is None:
        plt.show()
    else:
        plt.savefig(output_filepath + output_file_extension)
    plt.close(fig)


def plot_accuracy_per_difficulty_for_different_role_played_levels(
        avg_accuracy_per_grade_per_model,
        role_played_levels,
        dataset_name,
        prompt_idx,
        output_filepath: str = None,
        figsize: Optional[Tuple[int, int]] = None,
        output_file_extension: str = '.png',
):
    difficulty_levels = list(avg_accuracy_per_grade_per_model.keys())
    n_role_played_levels = len(role_played_levels)

    if figsize is None:
        figsize = (n_role_played_levels*2.5, 5)
    fig, ax = plt.subplots(1, n_role_played_levels, figsize=figsize, sharey='all')
    for idx, role_played_level in enumerate(role_played_levels):
        ax[idx].set_ylim(0, 1.0)
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].grid(alpha=0.5, axis='y')
        ax[idx].bar(difficulty_levels, [avg_accuracy_per_grade_per_model[i][idx] for i in difficulty_levels], color=COLORS[idx])
        ax[idx].set_title(f'{dataset_name} | prompt {prompt_idx}\n Role-played level: {role_played_level}')
        ax[idx].set_xlabel('Grade')
        ax[idx].set_ylabel('QA accuracy')
        ax[idx].set_xticks(difficulty_levels)
        ax[idx].set_xticklabels(difficulty_levels)
    if output_filepath is None:
        plt.show()
    else:
        plt.savefig(output_filepath + output_file_extension)
    plt.close(fig)


def get_all_info_for_plotting_by_mdoel_prompt_and_dataset(model, prompt_idx, dataset, split, complete_df, difficulty_column):
    data_path = os.path.join(OUTPUT_DATA_DIR, split, f'{model}_responses_{dataset}')
    student_levels = get_student_levels_from_prompt_idx(prompt_idx)
    filenames = [f"{model}_grade_answers_prompt{prompt_idx}_0shot_a_{1+idx}.csv" for idx, _ in enumerate(student_levels)]
    list_dfs = [pd.read_csv(os.path.join(data_path, filename)) for filename in filenames]
    # to keep only the questions that are answered by all role-played levels
    set_q_ids = get_questions_answered_by_all_roleplayed_levels(list_dfs, complete_df)
    avg_accuracy_per_model, avg_accuracy_per_grade_per_model = get_average_accuracy_per_model(
        list_dfs, set_q_ids, complete_df, difficulty_column)
    correctness_per_model, answers_per_model = get_response_correctness_per_model(list_dfs, set_q_ids, complete_df)
    return {
        'student_levels': student_levels,
        'avg_accuracy_per_model': avg_accuracy_per_model,
        'avg_accuracy_per_grade_per_model': avg_accuracy_per_grade_per_model,
        'correctness_per_model': correctness_per_model,
        'answers_per_model': answers_per_model,
    }

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utils import (
    get_student_levels_from_prompt_idx,
    get_original_dataset,
    get_questions_answered_by_all_roleplayed_levels,
    get_difficulty_dict_from_df,
    get_average_accuracy_per_model,
    get_response_correctness_per_model,
)
from utils_plotting import get_all_info_for_plotting_by_mdoel_prompt_and_dataset
from constants import (
    GPT_3_5,
    GPT_3_5_1106,
    OUTPUT_DATA_DIR,
    RACE,
    CUPA,
    ARC,
    DIFFICULTY_LEVELS,
)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


def main():
    out_fig_path = os.path.join('output_figures', 'for_paper')

    # SETUP -- info about datasets
    difficulty_levels_race = DIFFICULTY_LEVELS[RACE]
    difficulty_levels_cupa = DIFFICULTY_LEVELS[CUPA]
    difficulty_levels_arc = DIFFICULTY_LEVELS[ARC]

    complete_df_race = get_original_dataset(RACE)
    complete_df_cupa = get_original_dataset(CUPA)
    complete_df_arc = get_original_dataset(ARC)

    # dict that maps from qid to "true" difficulty
    difficulty_dict = get_difficulty_dict_from_df(complete_df_race)
    difficulty_dict = get_difficulty_dict_from_df(complete_df_cupa)
    difficulty_dict = get_difficulty_dict_from_df(complete_df_arc)

    # Simulation results
    dict_gpt_3_5_race_40 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 40, RACE, complete_df_race, difficulty_levels_race)
    dict_gpt_3_5_cupa_40 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 40, CUPA, complete_df_cupa, difficulty_levels_cupa)
    dict_gpt_3_5_arc_48 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 48, ARC, complete_df_arc, difficulty_levels_arc)
    dict_gpt_3_5_1106_race_40 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5_1106, 40, RACE, complete_df_race, difficulty_levels_race)
    # TODO: I don't have the results for this one below yet.
    # dict_gpt_3_5_1106_cupa_40 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5_1106, 40, CUPA, complete_df_cupa, difficulty_levels_cupa)
    dict_gpt_3_5_1106_arc_48 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5_1106, 48, ARC, complete_df_arc, difficulty_levels_arc)
    dict_gpt_3_5_race_44 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 44, RACE, complete_df_race, difficulty_levels_race)
    dict_gpt_3_5_race_45 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 45, RACE, complete_df_race, difficulty_levels_race)
    dict_gpt_3_5_race_46 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 46, RACE, complete_df_race, difficulty_levels_race)
    dict_gpt_3_5_arc_56 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 56, ARC, complete_df_arc, difficulty_levels_arc)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompt -- RACE and CUPA
    n_role_played_levels = len(dict_gpt_3_5_race_40['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_40['avg_accuracy_per_model'], 'o-', label='RACE', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40['avg_accuracy_per_model'], 'x-', label='CUPA', c='#b83266')
    ax.grid(alpha=0.5, axis='both')
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_race_40['student_levels'])
    ax.legend()
    # plt.show()
    # plt.savefig(os.path.join(out_fig_path, f'prompt_40_race_cupa_gpt_3_5_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompt -- ARC
    n_role_played_levels = len(dict_gpt_3_5_arc_48['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48['avg_accuracy_per_model'], '*-', label='ARC', c='#054b7d')
    ax.grid(alpha=0.5, axis='both')
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_arc_48['student_levels'])
    ax.legend()
    # plt.show()
    # plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_gpt_3_5_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level, separately for different question levels -- RACE
    label_idx_to_str = ['middle', 'high', 'college']
    difficulty_levels = list(dict_gpt_3_5_race_40['avg_accuracy_per_grade_per_model'].keys())
    n_role_played_levels = len(dict_gpt_3_5_race_40['student_levels'])
    fig, ax = plt.subplots(1, len(difficulty_levels), figsize=(7, 4.2), sharey='all')
    for idx, grade in enumerate(difficulty_levels):
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].grid(alpha=0.5, axis='y')
        ax[idx].plot(range(n_role_played_levels), dict_gpt_3_5_race_40['avg_accuracy_per_grade_per_model'][grade], 'o-', color='#ffab00')
        ax[idx].set_title(f'Q. level = "{label_idx_to_str[grade]}"')
        ax[idx].set_xlabel('Role-played level')
        ax[idx].set_ylabel('QA accuracy')
        ax[idx].set_xticks(range(n_role_played_levels))
        ax[idx].set_xticklabels(dict_gpt_3_5_race_40['student_levels'])
        ax[idx].set_ylim(0.25, 0.95)
    # plt.show()
    # plt.savefig(os.path.join(out_fig_path, f'prompt_40_race_gpt_3_5_mcqa_accuracy_per_level_by_question_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level, separately for different question levels -- ARC
    # label_idx_to_str = ['middle', 'high', 'college']
    difficulty_levels = list(dict_gpt_3_5_arc_48['avg_accuracy_per_grade_per_model'].keys())
    n_role_played_levels = len(dict_gpt_3_5_arc_48['student_levels'])
    fig, ax = plt.subplots(1, len(difficulty_levels), figsize=(14, 4.2), sharey='all')
    for idx, grade in enumerate(difficulty_levels):
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].grid(alpha=0.5, axis='y')
        ax[idx].plot(range(n_role_played_levels), dict_gpt_3_5_arc_48['avg_accuracy_per_grade_per_model'][grade], '*-', color='#054b7d')
        ax[idx].set_title(f'Q. level = "{grade}"')
        ax[idx].set_xlabel('Role-played level')
        if idx == 0:
            ax[idx].set_ylabel('MCQA accuracy')
        ax[idx].set_xticks(range(n_role_played_levels))
        ax[idx].set_xticklabels(dict_gpt_3_5_arc_48['student_levels'])
    # plt.show()
    plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_gpt_3_5_mcqa_accuracy_per_level_by_question_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompts, GPT_3_5 vs. GPT_3_5_1106 -- RACE, CUPA, ARC
    n_role_played_levels = len(dict_gpt_3_5_1106_arc_48['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_1106_arc_48['avg_accuracy_per_model'], '*-', label='ARC (GPT-3.5 v1106)', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48['avg_accuracy_per_model'], '*:', label='ARC (GPT-3.5)', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_1106_race_40['avg_accuracy_per_model'], 'o-', label='RACE (GPT-3.5 v1106)', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_40['avg_accuracy_per_model'], 'o:', label='RACE (GPT-3.5)', c='#ffab00')
    # ax.plot(range(n_role_played_levels), dict_gpt_3_5_1106_cupa_40['avg_accuracy_per_model'], 'x-', label='CUPA', c='#b83266')  # TODO: I don't have these results yet!
    ax.grid(alpha=0.5, axis='both')
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_1106_arc_48['student_levels'])
    ax.set_ylim(0.38, 0.92)
    ax.legend()
    # plt.show()
    # plt.savefig(os.path.join(out_fig_path, f'prompt_48_40_gpt_3_5_vs_gpt_3_5_1106_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- language proficiency scales -- RACE
    # TODO: all this on CUPA
    labels = ['CEFR', 'IELTS', 'TOEFL']
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(7, 4.2))
    for idx, dict_results in enumerate([dict_gpt_3_5_race_44, dict_gpt_3_5_race_45, dict_gpt_3_5_race_46]):
        n_role_played_levels = len(dict_results['student_levels'])
        ax[idx].plot(range(n_role_played_levels), dict_results['avg_accuracy_per_model'], '*:', label=labels[idx], c='#054b7d')
        # ax[idx].plot(range(n_role_played_levels), dict_gpt_3_5_race_40['avg_accuracy_per_model'], 'o:', label='Ref. prompt', c='#ffab00')
        ax[idx].grid(alpha=0.5, axis='both')
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        if idx == 0:
            ax[idx].set_ylabel('MCQA accuracy')
        ax[idx].set_xlabel('Simulated level')
        ax[idx].set_xticks(range(n_role_played_levels))
        if idx > 0:
            ax[idx].set_xticklabels([x if x_idx%2 == 0 else '' for x_idx, x in enumerate(dict_results['student_levels'])])
        else:
            ax[idx].set_xticklabels(dict_results['student_levels'])
        ax[idx].set_ylim(0.38, 0.92)
        ax[idx].legend()
    # plt.show()
    # plt.savefig(os.path.join(out_fig_path, f'prompt_44_45_46_gpt_3_5_race_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompts, GPT_3_5 vs. GPT_3_5_1106 -- RACE, CUPA, ARC
    n_role_played_levels = len(dict_gpt_3_5_arc_56['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_56['avg_accuracy_per_model'], '*-', label='ARC', c='#054b7d')
    ax.grid(alpha=0.5, axis='both')
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels([x if x_idx % 2 == 0 else '' for x_idx, x in enumerate(dict_gpt_3_5_arc_56['student_levels'])])
    ax.set_ylim(0.38, 0.92)
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(out_fig_path, f'prompt_56_gpt_3_5_arc_1106_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# saved colours:
# RACE: #ffab00 (also tried #fd7401)
# ARC: #054b7d
# CUPA: #b83266

if __name__ == "__main__":
    main()

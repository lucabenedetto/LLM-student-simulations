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
)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

DO_PLOT = False
SAVE_FIG = False


def main():
    out_fig_path = os.path.join('output_figures', 'for_paper')

    # SETUP -- info about datasets
    difficulty_column_race = 'difficulty'
    difficulty_column_cupa = 'target_level'
    difficulty_column_arc = 'difficulty'

    complete_df_race = get_original_dataset(RACE)
    complete_df_cupa = get_original_dataset(CUPA)
    complete_df_arc = get_original_dataset(ARC)

    # # dict that maps from qid to "true" difficulty
    # difficulty_dict = get_difficulty_dict_from_df(complete_df_race)
    difficulty_dict_cupa = get_difficulty_dict_from_df(complete_df_cupa)
    target_level_dict_cupa = get_difficulty_dict_from_df(complete_df_cupa, difficulty_column='target_level')
    # difficulty_dict = get_difficulty_dict_from_df(complete_df_arc)

    # Simulation results
    dict_gpt_3_5_race_40 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 40, RACE, complete_df_race, difficulty_column_race)
    dict_gpt_3_5_cupa_40 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 40, CUPA, complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_arc_48 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 48, ARC, complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_1106_race_40 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5_1106, 40, RACE, complete_df_race, difficulty_column_race)
    dict_gpt_3_5_1106_cupa_40 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5_1106, 40, CUPA, complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_1106_arc_48 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5_1106, 48, ARC, complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_race_44 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 44, RACE, complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_45 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 45, RACE, complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_46 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 46, RACE, complete_df_race, difficulty_column_race)
    dict_gpt_3_5_cupa_44 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 44, CUPA, complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_45 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 45, CUPA, complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_46 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 46, CUPA, complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_arc_56 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 56, ARC, complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_arc_55 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 55, ARC, complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_race_57 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 57, RACE, complete_df_race, difficulty_column_race)
    dict_gpt_3_5_cupa_57 = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 57, CUPA, complete_df_cupa, difficulty_column_cupa)


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
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_40_race_cupa_gpt_3_5_mcqa_accuracy_per_level.pdf'))
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
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_gpt_3_5_mcqa_accuracy_per_level.pdf'))
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
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_40_race_gpt_3_5_mcqa_accuracy_per_level_by_question_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level, separately for different question levels -- ARC
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
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_gpt_3_5_mcqa_accuracy_per_level_by_question_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level, separately for different target levels -- CUPA
    difficulty_levels = list(dict_gpt_3_5_cupa_40['avg_accuracy_per_grade_per_model'].keys())
    n_role_played_levels = len(dict_gpt_3_5_cupa_40['student_levels'])
    fig, ax = plt.subplots(1, len(difficulty_levels), figsize=(10, 4.2), sharey='all')
    for idx, grade in enumerate(difficulty_levels):
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].grid(alpha=0.5, axis='y')
        ax[idx].plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40['avg_accuracy_per_grade_per_model'][grade], 'x-', color='#b83266')
        ax[idx].set_title(f'Target level = "{grade}"')
        ax[idx].set_xlabel('Role-played level')
        ax[idx].set_ylabel('QA accuracy')
        ax[idx].set_xticks(range(n_role_played_levels))
        ax[idx].set_xticklabels(dict_gpt_3_5_cupa_40['student_levels'])
        ax[idx].set_ylim(0.25, 0.95)
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_40_cupa_gpt_3_5_mcqa_accuracy_per_level_by_question_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: Correlation between pretested difficulty and virtual pretesting -- CUPA
    #   The first plot plots all the questions, the second one only the ones which have correctness != 1.0 and != 0.0
    for plot_idx in [0, 1]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.grid(alpha=0.5, axis='y')
        X, Y = [], []
        for q_id in dict_gpt_3_5_cupa_40['correctness_per_model'].keys():
            if plot_idx == 0 or np.mean(dict_gpt_3_5_cupa_40['correctness_per_model'][q_id]) not in {0.0, 1.0}:
                X.append(difficulty_dict_cupa[q_id])
                Y.append(np.mean(dict_gpt_3_5_cupa_40['correctness_per_model'][q_id]))
        ax.scatter(X, Y, color='#b83266')
        m, b = np.polyfit(X, Y, 1)
        if m and b:
            x0, x1 = min(X), max(X)
            ax.plot([x0, x1], [x0 * m + b, x1 * m + b], ':', c='k', label='linear fit')
        ax.set_xlabel('True difficulty')
        ax.set_ylabel('Difficulty from virtual pretesting')
        ax.legend()
        if DO_PLOT: plt.show()
        if plot_idx == 0:
            output_name = f'prompt_40_cupa_gpt_3_5_virtual_pretesting.pdf'
        else:
            output_name = f'prompt_40_cupa_gpt_3_5_virtual_pretesting_no_extremes.pdf'
        if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, output_name))
        plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: Correlation between pretested difficulty and virtual pretesting, separately for target level -- CUPA
    #   The first plot plots all the questions, the second one only the ones which have correctness != 1.0 and != 0.0
    for plot_idx in [0, 1]:
        X = {'B1': [], 'B2': [], 'C1': [], 'C2': []}
        Y = {'B1': [], 'B2': [], 'C1': [], 'C2': []}
        for q_id in dict_gpt_3_5_cupa_40['correctness_per_model'].keys():
            if plot_idx == 0 or np.mean(dict_gpt_3_5_cupa_40['correctness_per_model'][q_id]) not in {0.0, 1.0}:
                X[target_level_dict_cupa[q_id]].append(difficulty_dict_cupa[q_id])
                Y[target_level_dict_cupa[q_id]].append(np.mean(dict_gpt_3_5_cupa_40['correctness_per_model'][q_id]))
        fig, ax = plt.subplots(1, 4, figsize=(14, 4.2), sharex=True, sharey=True)
        for idx, target_level in enumerate(['B1', 'B2', 'C1', 'C2']):
            ax[idx].scatter(X[target_level], Y[target_level], color='#b83266', label=target_level)
            m, b = np.polyfit(X[target_level], Y[target_level], 1)
            if m and b:
                x0, x1 = 30, 110
                ax[idx].plot([x0, x1], [x0 * m + b, x1 * m + b], ':', c='k', label='linear fit')
            # ax[idx].set_title(target_level)
            ax[idx].set_xlabel('True difficulty')
            ax[idx].set_xticks(range(30, 111, 10))
            ax[idx].grid(alpha=0.5, axis='both')
            ax[idx].legend()
        ax[0].set_ylabel('Difficulty from virtual pretesting')
        if DO_PLOT: plt.show()
        if plot_idx == 0:
            output_name = f'prompt_40_cupa_gpt_3_5_virtual_pretesting_by_target_level.pdf'
        else:
            output_name = f'prompt_40_cupa_gpt_3_5_virtual_pretesting_by_target_level_no_extremes.pdf'
        if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, output_name))
        plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompts, GPT_3_5 vs. GPT_3_5_1106 -- RACE, CUPA, ARC
    n_role_played_levels = len(dict_gpt_3_5_1106_arc_48['student_levels'])

    # fig, ax = plt.subplots(figsize=(6, 4.2))  # this was when I had all the models in one plot.
    fig, ax = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    ax[0].plot(range(n_role_played_levels), dict_gpt_3_5_1106_arc_48['avg_accuracy_per_model'], '*-', label='ARC (GPT-3.5 v1106)', c='#054b7d')
    ax[0].plot(range(n_role_played_levels), dict_gpt_3_5_arc_48['avg_accuracy_per_model'], '*:', label='ARC (GPT-3.5)', c='#054b7d')
    ax[1].plot(range(n_role_played_levels), dict_gpt_3_5_1106_race_40['avg_accuracy_per_model'], 'o-', label='RACE (GPT-3.5 v1106)', c='#ffab00')
    ax[1].plot(range(n_role_played_levels), dict_gpt_3_5_race_40['avg_accuracy_per_model'], 'o:', label='RACE (GPT-3.5)', c='#ffab00')
    ax[2].plot(range(n_role_played_levels), dict_gpt_3_5_1106_cupa_40['avg_accuracy_per_model'], 'x-', label='CUPA (GPT-3.5 v1106)', c='#b83266')
    ax[2].plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40['avg_accuracy_per_model'], 'x:', label='CUPA', c='#b83266')
    for idx in range(3):
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].set_ylabel('MCQA accuracy')
        ax[idx].set_xticks(range(n_role_played_levels))
        ax[idx].set_xticklabels(dict_gpt_3_5_1106_arc_48['student_levels'])
        ax[idx].set_ylim(0.38, 0.92)
        ax[idx].grid(alpha=0.5, axis='both')
        ax[idx].legend()
    ax[0].set_xlabel('Simulated level')
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_40_gpt_3_5_vs_gpt_3_5_1106_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- language proficiency scales -- RACE and CUPA
    labels = ['CEFR', 'IELTS', 'TOEFL']
    fig, ax = plt.subplots(3, 1, figsize=(6, 9))
    results_race = [dict_gpt_3_5_race_44, dict_gpt_3_5_race_45, dict_gpt_3_5_race_46]
    results_cupa = [dict_gpt_3_5_cupa_44, dict_gpt_3_5_cupa_45, dict_gpt_3_5_cupa_46]
    for idx in range(3):
        n_role_played_levels = len(results_race[idx]['student_levels'])
        ax[idx].plot(range(n_role_played_levels), results_race[idx]['avg_accuracy_per_model'], 'o:', label=f'RACE - {labels[idx]}', c='#ffab00')
        ax[idx].plot(range(n_role_played_levels), results_cupa[idx]['avg_accuracy_per_model'], 'x:', label=f'CUPA - {labels[idx]}', c='#b83266')
        ax[idx].grid(alpha=0.5, axis='both')
        ax[idx].set_yticks(np.arange(0.4, 1.0, 0.1))
        ax[idx].set_ylabel('MCQA accuracy')
        ax[idx].set_xticks(range(n_role_played_levels))
        # ax[idx].set_xticklabels([x if x_idx%2 == 0 else '' for x_idx, x in enumerate(results_race[idx]['student_levels'])])
        ax[idx].set_xticklabels(results_race[idx]['student_levels'])
        # ax[idx].set_ylim(0.38, 0.92)
        ax[idx].legend()
    ax[2].set_xlabel('Simulated level')
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_44_45_46_gpt_3_5_race_mcqa_accuracy_per_level.pdf'))
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
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_56_gpt_3_5_arc_1106_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- exam grades -- ARC, RACE, and CUPA
    n_role_played_levels = len(dict_gpt_3_5_race_57['student_levels'])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_57['avg_accuracy_per_model'], 'o-', label='RACE', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_55['avg_accuracy_per_model'], '*-', label='ARC', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_cupa_57['avg_accuracy_per_model'], 'x-', label='CUPA', c='#b83266')
    ax.grid(alpha=0.5, axis='both')
    # ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_race_57['student_levels'])
    ax.legend()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_57_race_cupa_55_arc_gpt_3_5_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# saved colours:
# RACE: #ffab00 (also tried #fd7401)
# ARC: #054b7d
# CUPA: #b83266

if __name__ == "__main__":
    main()

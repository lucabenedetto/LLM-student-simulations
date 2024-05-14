import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

from utils import (
    get_original_dataset,
    get_difficulty_dict_from_df,
    item_response_function as irf,
)
from utils_plotting import get_all_info_for_plotting_by_mdoel_prompt_and_dataset
from constants import (
    GPT_3_5,
    GPT_3_5_1106,
    GPT_4_1106,
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

    # dict that maps from qid to "true" difficulty
    difficulty_dict_cupa = get_difficulty_dict_from_df(complete_df_cupa)
    target_level_dict_cupa = get_difficulty_dict_from_df(complete_df_cupa, difficulty_column='target_level')

    # Simulation results

    # GPT-3.5
    dict_gpt_3_5_arc_28_dev = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 28, ARC, 'dev', complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_arc_31_dev = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 31, ARC, 'dev', complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_arc_32_dev = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 32, ARC, 'dev', complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_arc_35_dev = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 35, ARC, 'dev', complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_arc_48_dev = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 48, ARC, 'dev', complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_arc_48_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 48, ARC, 'test', complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_arc_52_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 52, ARC, 'test', complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_arc_55_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 55, ARC, 'test', complete_df_arc, difficulty_column_arc)
    dict_gpt_3_5_arc_56_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 56, ARC, 'test', complete_df_arc, difficulty_column_arc)

    dict_gpt_3_5_race_40_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 40, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_44_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 44, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_45_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 45, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_46_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 46, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_47_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 47, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_49_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 49, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_50_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 50, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_53_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 53, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_57_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 57, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_race_60_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 60, RACE, 'test', complete_df_race, difficulty_column_race)

    dict_gpt_3_5_cupa_40_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 40, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_44_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 44, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_45_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 45, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_46_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 46, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_47_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 47, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_49_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 49, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_50_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 50, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_53_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 53, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_57_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 57, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_cupa_60_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5, 60, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)

    # GPT-3.5 1106
    dict_gpt_3_5_1106_race_40_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5_1106, 40, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_3_5_1106_cupa_40_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5_1106, 40, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)
    dict_gpt_3_5_1106_arc_48_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_3_5_1106, 48, ARC, 'test', complete_df_arc, difficulty_column_arc)

    # GPT-4
    dict_gpt_4_1106_arc_48_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_4_1106, 48, ARC, 'test', complete_df_arc, difficulty_column_arc)
    dict_gpt_4_1106_race_40_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_4_1106, 40, RACE, 'test', complete_df_race, difficulty_column_race)
    dict_gpt_4_1106_cupa_40_test = get_all_info_for_plotting_by_mdoel_prompt_and_dataset(GPT_4_1106, 40, CUPA, 'test', complete_df_cupa, difficulty_column_cupa)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompt -- RACE and CUPA
    n_role_played_levels = len(dict_gpt_3_5_race_40_test['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_40_test['avg_accuracy_per_model'], 'o-', label='RACE', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40_test['avg_accuracy_per_model'], 'x-', label='CUPA', c='#b83266')
    ax.grid(alpha=0.5, axis='both')
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.0, 0.05))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_race_40_test['student_levels'])
    ax.legend()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_40_race_cupa_gpt_3_5_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompt -- ARC dev and test
    n_role_played_levels = len(dict_gpt_3_5_arc_48_test['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_test['avg_accuracy_per_model'], '*-', label='ARC (test)', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_dev['avg_accuracy_per_model'], '*:', label='ARC (dev)', c='#054b7d')
    ax.grid(alpha=0.5, axis='both')
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.0, 0.05))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_arc_48_test['student_levels'])
    ax.legend()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_gpt_3_5_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompt -- ARC (test), RACE and CUPA
    n_role_played_levels = len(dict_gpt_3_5_arc_48_test['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_test['avg_accuracy_per_model'], '*-', label='ARC (test)', c='#054b7d')
    # ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_dev['avg_accuracy_per_model'], '*:', label='ARC (dev)', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_40_test['avg_accuracy_per_model'], 'o-', label='RACE', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40_test['avg_accuracy_per_model'], 'x-', label='CUPA', c='#b83266')
    ax.grid(alpha=0.5, axis='both')
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.0, 0.05))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_arc_48_test['student_levels'])
    ax.legend()
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_40_race_cupa_gpt_3_5_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompt -- ARC (test), ARC (dev), RACE and CUPA
    n_role_played_levels = len(dict_gpt_3_5_arc_48_test['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_dev['avg_accuracy_per_model'], '*:', label='ARC (dev)', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_test['avg_accuracy_per_model'], '*-', label='ARC (test)', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_40_test['avg_accuracy_per_model'], 'o-', label='RACE', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40_test['avg_accuracy_per_model'], 'x-', label='CUPA', c='#b83266')
    ax.grid(alpha=0.5, axis='both')
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.0, 0.05))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_arc_48_test['student_levels'])
    ax.legend()
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_40_race_cupa_gpt_3_5_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level, separately for different question levels -- RACE
    label_idx_to_str = ['middle', 'high', 'college']
    plot_style = ['o-', 'o--', 'o:']
    difficulty_levels = list(dict_gpt_3_5_race_40_test['avg_accuracy_per_grade_per_model'].keys())
    n_role_played_levels = len(dict_gpt_3_5_race_40_test['student_levels'])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.set_yticks(np.arange(0.0, 1.0, 0.05))
    ax.grid(alpha=0.5)
    for idx, grade in enumerate(difficulty_levels):
        ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_40_test['avg_accuracy_per_grade_per_model'][grade], plot_style[idx], label=label_idx_to_str[idx], color='#ffab00')
    ax.set_xlabel('Simulated level')
    ax.set_ylabel('MCQA accuracy')
    ax.legend()
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_race_40_test['student_levels'])
    ax.set_ylim(0.25, 0.9)
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_40_race_gpt_3_5_mcqa_accuracy_per_level_by_question_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level, separately for different question levels -- ARC
    plot_style = ['*-', '', '*--', '', '*-.', '', '*:']
    difficulty_levels = list(dict_gpt_3_5_arc_48_test['avg_accuracy_per_grade_per_model'].keys())
    n_role_played_levels = len(dict_gpt_3_5_arc_48_test['student_levels'])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for idx, grade in enumerate(difficulty_levels):
        if (idx+1) % 2 == 0:
            continue
        ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_test['avg_accuracy_per_grade_per_model'][grade], plot_style[idx], label=grade, color='#054b7d')
    ax.set_yticks(np.arange(0.5, 1.0, 0.05))
    ax.grid(alpha=0.5)
    ax.set_xlabel('Simulated level')
    ax.set_ylabel('MCQA accuracy')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_arc_48_test['student_levels'])
    ax.legend()
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_gpt_3_5_mcqa_accuracy_per_level_by_question_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level, separately for different question levels -- ARC -- same as the previous one but
    #   on the other grades (even instead of odd numbers)
    plot_style = ['', '*-', '', '*--', '', '*:', '']
    difficulty_levels = list(dict_gpt_3_5_arc_48_test['avg_accuracy_per_grade_per_model'].keys())
    n_role_played_levels = len(dict_gpt_3_5_arc_48_test['student_levels'])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for idx, grade in enumerate(difficulty_levels):
        if (idx) % 2 == 0:
            continue
        ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_test['avg_accuracy_per_grade_per_model'][grade], plot_style[idx], label=grade, color='#054b7d')
    ax.set_yticks(np.arange(0.5, 1.0, 0.05))
    ax.grid(alpha=0.5)
    ax.set_xlabel('Simulated level')
    ax.set_ylabel('MCQA accuracy')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_arc_48_test['student_levels'])
    ax.legend()
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_gpt_3_5_mcqa_accuracy_per_level_by_question_level_b.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level, separately for different question levels -- CUPA
    n_role_played_levels = len(dict_gpt_3_5_cupa_40_test['student_levels'])
    q_id_level_0 = []
    q_id_level_1 = []
    q_id_level_2 = []
    for q_id in dict_gpt_3_5_cupa_40_test['correctness_per_model'].keys():
        if difficulty_dict_cupa[q_id] < 60:
            q_id_level_0.append(q_id)
        elif difficulty_dict_cupa[q_id] <= 80:
            q_id_level_1.append(q_id)
        else:  # > 80
            q_id_level_2.append(q_id)
    y_difficulty_0 = [
        np.mean([dict_gpt_3_5_cupa_40_test['correctness_per_model'][q_id][simulated_level] for q_id in q_id_level_0])
        for simulated_level in range(n_role_played_levels)
    ]
    y_difficulty_1 = [
        np.mean([dict_gpt_3_5_cupa_40_test['correctness_per_model'][q_id][simulated_level] for q_id in q_id_level_1])
        for simulated_level in range(n_role_played_levels)
    ]
    y_difficulty_2 = [
        np.mean([dict_gpt_3_5_cupa_40_test['correctness_per_model'][q_id][simulated_level] for q_id in q_id_level_2])
        for simulated_level in range(n_role_played_levels)
    ]
    label_idx_to_str = ['diff. < 60', '60 <= diff. <= 80', 'diff. > 80']
    plot_style = ['x-', 'x--', 'x:']
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.set_yticks(np.arange(0.0, 1.0, 0.05))
    ax.grid(alpha=0.5)
    ax.plot(range(n_role_played_levels), y_difficulty_0, plot_style[0], color='#b83266', label=label_idx_to_str[0])
    ax.plot(range(n_role_played_levels), y_difficulty_1, plot_style[1], color='#b83266', label=label_idx_to_str[1])
    ax.plot(range(n_role_played_levels), y_difficulty_2, plot_style[2], color='#b83266', label=label_idx_to_str[2])
    ax.set_xlabel('Simulated level')
    ax.set_ylabel('MCQA accuracy')
    ax.legend()
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_cupa_40_test['student_levels'])
    plt.tight_layout()
    # ax.set_ylim(0.25, 0.9)
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_40_cupa_gpt_3_5_mcqa_accuracy_per_level_by_question_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level, separately for different target levels -- CUPA
    difficulty_levels = list(dict_gpt_3_5_cupa_40_test['avg_accuracy_per_grade_per_model'].keys())
    n_role_played_levels = len(dict_gpt_3_5_cupa_40_test['student_levels'])
    fig, ax = plt.subplots(1, len(difficulty_levels), figsize=(10, 4.2), sharey='all')
    for idx, grade in enumerate(difficulty_levels):
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].grid(alpha=0.5, axis='y')
        ax[idx].plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40_test['avg_accuracy_per_grade_per_model'][grade], 'x-', color='#b83266')
        ax[idx].set_title(f'Target level = "{grade}"')
        ax[idx].set_xlabel('Role-played level')
        ax[idx].set_ylabel('QA accuracy')
        ax[idx].set_xticks(range(n_role_played_levels))
        ax[idx].set_xticklabels(dict_gpt_3_5_cupa_40_test['student_levels'])
        ax[idx].set_ylim(0.25, 0.95)
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_40_cupa_gpt_3_5_mcqa_accuracy_per_level_by_question_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: Correlation between pretested difficulty and virtual pretesting -- CUPA
    #   The first plot plots all the questions, the second one only the ones which have correctness != 1.0 and != 0.0
    for plot_idx in [0, 1]:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.grid(alpha=0.5, axis='y')
        X, Y = [], []
        for q_id in dict_gpt_3_5_cupa_40_test['correctness_per_model'].keys():
            if plot_idx == 0 or np.mean(dict_gpt_3_5_cupa_40_test['correctness_per_model'][q_id]) not in {0.0, 1.0}:
                X.append(difficulty_dict_cupa[q_id])
                Y.append(1-np.mean(dict_gpt_3_5_cupa_40_test['correctness_per_model'][q_id]))
        ax.scatter(X, [y + np.random.uniform(-0.025, 0.025) for y in Y], color='#b83266', alpha=0.75)
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
        if plot_idx == 0:
            np.random.seed(42)
            print("LLM diff:", scipy.stats.linregress(X, Y))
            random_difficulty = np.random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], len(Y))
            print("Random", scipy.stats.linregress(X, random_difficulty))

            # TODO: this is the simulation with IRT, possibly to move somewhere else.
            answers_sim_students = [[], [], [], [], []]
            q_ids = list(dict_gpt_3_5_cupa_40_test['correctness_per_model'].keys())
            for q_id in q_ids:
                # I need the 1- because I want the difficulty, not the accuracy
                answers_sim_students[0].append(1-irf(difficulty_dict_cupa[q_id], 30) >= np.random.uniform(0, 1.0))
                answers_sim_students[1].append(1-irf(difficulty_dict_cupa[q_id], 50) >= np.random.uniform(0, 1.0))
                answers_sim_students[2].append(1-irf(difficulty_dict_cupa[q_id], 70) >= np.random.uniform(0, 1.0))
                answers_sim_students[3].append(1-irf(difficulty_dict_cupa[q_id], 90) >= np.random.uniform(0, 1.0))
                answers_sim_students[4].append(1-irf(difficulty_dict_cupa[q_id], 110) >= np.random.uniform(0, 1.0))
            simulated_students_answers = np.average(answers_sim_students, axis=0)
            print(len(simulated_students_answers) == len(q_ids))
            difficulties = [difficulty_dict_cupa[q_id] for q_id in q_ids]
            print("Simulated students (ideal):", scipy.stats.linregress(difficulties, simulated_students_answers))

            answers_sim_students = [[], [], [], [], []]
            q_ids = list(dict_gpt_3_5_cupa_40_test['correctness_per_model'].keys())
            for q_id in q_ids:
                # I need the 1- because I want the difficulty, not the accuracy
                answers_sim_students[0].append(1-irf(difficulty_dict_cupa[q_id], 80) >= np.random.uniform(0, 1.0))
                answers_sim_students[1].append(1-irf(difficulty_dict_cupa[q_id], 90) >= np.random.uniform(0, 1.0))
                answers_sim_students[2].append(1-irf(difficulty_dict_cupa[q_id], 100) >= np.random.uniform(0, 1.0))
                answers_sim_students[3].append(1-irf(difficulty_dict_cupa[q_id], 110) >= np.random.uniform(0, 1.0))
                answers_sim_students[4].append(1-irf(difficulty_dict_cupa[q_id], 120) >= np.random.uniform(0, 1.0))
            simulated_students_answers = np.average(answers_sim_students, axis=0)
            print(len(simulated_students_answers) == len(q_ids))
            difficulties = [difficulty_dict_cupa[q_id] for q_id in q_ids]
            print("Simulated students (not ideal):", scipy.stats.linregress(difficulties, simulated_students_answers))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: Correlation between pretested difficulty and virtual pretesting, separately for target level -- CUPA
    #   The first plot plots all the questions, the second one only the ones which have correctness != 1.0 and != 0.0
    for plot_idx in [0, 1]:
        X = {'B1': [], 'B2': [], 'C1': [], 'C2': []}
        Y = {'B1': [], 'B2': [], 'C1': [], 'C2': []}
        for q_id in dict_gpt_3_5_cupa_40_test['correctness_per_model'].keys():
            if plot_idx == 0 or np.mean(dict_gpt_3_5_cupa_40_test['correctness_per_model'][q_id]) not in {0.0, 1.0}:
                X[target_level_dict_cupa[q_id]].append(difficulty_dict_cupa[q_id])
                Y[target_level_dict_cupa[q_id]].append(1-np.mean(dict_gpt_3_5_cupa_40_test['correctness_per_model'][q_id]))
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
    # FIGURE: average accuracy per model -- reference prompts, GPT_3_5 vs. GPT_3_5_1106 vs. GPT-4 -- RACE, CUPA, ARC
    # (in three different plots)
    # ARC
    n_role_played_levels = len(dict_gpt_3_5_1106_arc_48_test['student_levels'])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_test['avg_accuracy_per_model'], '*-', label='GPT-3.5', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_1106_arc_48_test['avg_accuracy_per_model'], '*--', label='GPT-3.5 v1106', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_4_1106_arc_48_test['avg_accuracy_per_model'], '*:', label='GPT-4', c='#054b7d')
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_1106_arc_48_test['student_levels'])
    ax.set_ylim(0.38, 1.02)
    ax.grid(alpha=0.5, axis='both')
    ax.legend()
    ax.set_xlabel('Simulated level')
    ax.set_ylabel('MCQA accuracy')
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_arc_gpt_3_5_vs_gpt_3_5_1106_vs_gpt_4_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # RACE
    n_role_played_levels = len(dict_gpt_3_5_1106_arc_48_test['student_levels'])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_40_test['avg_accuracy_per_model'], 'o-', label='GPT-3.5', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_1106_race_40_test['avg_accuracy_per_model'], 'o--', label='GPT-3.5 v1106', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_4_1106_race_40_test['avg_accuracy_per_model'], 'o:', label='GPT-4', c='#ffab00')
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_1106_arc_48_test['student_levels'])
    ax.set_ylim(0.38, 1.02)
    ax.grid(alpha=0.5, axis='both')
    ax.legend()
    ax.set_xlabel('Simulated level')
    ax.set_ylabel('MCQA accuracy')
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_40_race_gpt_3_5_vs_gpt_3_5_1106_vs_gpt_4_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CUPA
    n_role_played_levels = len(dict_gpt_3_5_1106_arc_48_test['student_levels'])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40_test['avg_accuracy_per_model'], 'x-', label='GPT-3.5', c='#b83266')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_1106_cupa_40_test['avg_accuracy_per_model'], 'x--', label='GPT-3.5 v1106', c='#b83266')
    ax.plot(range(n_role_played_levels), dict_gpt_4_1106_cupa_40_test['avg_accuracy_per_model'], 'x:', label='GPT-4', c='#b83266')
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_1106_arc_48_test['student_levels'])
    ax.set_ylim(0.38, 1.02)
    ax.grid(alpha=0.5, axis='both')
    ax.legend()
    ax.set_xlabel('Simulated level')
    ax.set_ylabel('MCQA accuracy')
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_40_cupa_gpt_3_5_vs_gpt_3_5_1106_vs_gpt_4_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompts, GPT_3_5 vs. GPT_3_5_1106 -- RACE, CUPA, ARC
    n_role_played_levels = len(dict_gpt_3_5_1106_arc_48_test['student_levels'])

    # fig, ax = plt.subplots(figsize=(6, 4.2))  # this was when I had all the models in one plot.
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharex=True, sharey=True)
    ax[0].plot(range(n_role_played_levels), dict_gpt_3_5_1106_arc_48_test['avg_accuracy_per_model'], '*-', label='ARC (GPT-3.5 v1106)', c='#054b7d')
    ax[0].plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_test['avg_accuracy_per_model'], '*:', label='ARC (GPT-3.5)', c='#054b7d')
    ax[1].plot(range(n_role_played_levels), dict_gpt_3_5_1106_race_40_test['avg_accuracy_per_model'], 'o-', label='RACE (GPT-3.5 v1106)', c='#ffab00')
    ax[1].plot(range(n_role_played_levels), dict_gpt_3_5_race_40_test['avg_accuracy_per_model'], 'o:', label='RACE (GPT-3.5)', c='#ffab00')
    ax[2].plot(range(n_role_played_levels), dict_gpt_3_5_1106_cupa_40_test['avg_accuracy_per_model'], 'x-', label='CUPA (GPT-3.5 v1106)', c='#b83266')
    ax[2].plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40_test['avg_accuracy_per_model'], 'x:', label='CUPA', c='#b83266')
    for idx in range(3):
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].set_xticks(range(n_role_played_levels))
        ax[idx].set_xticklabels(dict_gpt_3_5_1106_arc_48_test['student_levels'])
        ax[idx].set_ylim(0.38, 0.92)
        ax[idx].grid(alpha=0.5, axis='both')
        ax[idx].legend()
        ax[idx].set_xlabel('Simulated level')
    ax[0].set_ylabel('MCQA accuracy')
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_40_gpt_3_5_vs_gpt_3_5_1106_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompts, GPT_3_5 vs. GPT_3_5_1106 -- RACE, CUPA, ARC
    # It is the same as the previous plot, but plotting them in separate figs.
    # TODO
    n_role_played_levels = len(dict_gpt_3_5_1106_arc_48_test['student_levels'])

    # fig, ax = plt.subplots(figsize=(6, 4.2))  # this was when I had all the models in one plot.
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharex=True, sharey=True)
    ax[0].plot(range(n_role_played_levels), dict_gpt_3_5_1106_arc_48_test['avg_accuracy_per_model'], '*-', label='ARC (GPT-3.5 v1106)', c='#054b7d')
    ax[0].plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_test['avg_accuracy_per_model'], '*:', label='ARC (GPT-3.5)', c='#054b7d')
    ax[1].plot(range(n_role_played_levels), dict_gpt_3_5_1106_race_40_test['avg_accuracy_per_model'], 'o-', label='RACE (GPT-3.5 v1106)', c='#ffab00')
    ax[1].plot(range(n_role_played_levels), dict_gpt_3_5_race_40_test['avg_accuracy_per_model'], 'o:', label='RACE (GPT-3.5)', c='#ffab00')
    ax[2].plot(range(n_role_played_levels), dict_gpt_3_5_1106_cupa_40_test['avg_accuracy_per_model'], 'x-', label='CUPA (GPT-3.5 v1106)', c='#b83266')
    ax[2].plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40_test['avg_accuracy_per_model'], 'x:', label='CUPA', c='#b83266')
    for idx in range(3):
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].set_xticks(range(n_role_played_levels))
        ax[idx].set_xticklabels(dict_gpt_3_5_1106_arc_48_test['student_levels'])
        ax[idx].set_ylim(0.38, 0.92)
        ax[idx].grid(alpha=0.5, axis='both')
        ax[idx].legend()
        ax[idx].set_xlabel('Simulated level')
    ax[0].set_ylabel('MCQA accuracy')
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_40_gpt_3_5_vs_gpt_3_5_1106_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompts, GPT_3_5 vs. GPT_4_1106 -- ARC, RACE, and CUPA
    n_role_played_levels = len(dict_gpt_3_5_arc_48_test['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(n_role_played_levels), dict_gpt_4_1106_arc_48_test['avg_accuracy_per_model'], '*-', label='ARC', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_48_test['avg_accuracy_per_model'], '*:', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_4_1106_race_40_test['avg_accuracy_per_model'], 'o-', label='RACE', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_40_test['avg_accuracy_per_model'], 'o:', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_4_1106_cupa_40_test['avg_accuracy_per_model'], 'x-', label='CUPA', c='#b83266')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_cupa_40_test['avg_accuracy_per_model'], 'x:', c='#b83266')
    ax.set_yticks(np.arange(0.3, 1.0, 0.05))
    ax.set_yticklabels([f'%.2f' % i if idx % 2 == 1 else '' for idx, i in enumerate(np.arange(0.3, 1.0, 0.05))])
    ax.set_ylabel('MCQA accuracy')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_1106_arc_48_test['student_levels'])
    ax.grid(alpha=0.5, axis='both')
    ax.legend()
    ax.set_xlabel('Simulated level')
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_48_40_gpt_3_5_vs_gpt_4_1106_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- language proficiency scales -- RACE and CUPA
    labels = ['CEFR', 'IELTS', 'TOEFL']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    results_race = [dict_gpt_3_5_race_44_test, dict_gpt_3_5_race_45_test, dict_gpt_3_5_race_46_test]
    results_cupa = [dict_gpt_3_5_cupa_44_test, dict_gpt_3_5_cupa_45_test, dict_gpt_3_5_cupa_46_test]
    for idx in range(3):
        n_role_played_levels = len(results_race[idx]['student_levels'])
        ax[idx].plot(range(n_role_played_levels), results_race[idx]['avg_accuracy_per_model'], 'o:', label=f'RACE - {labels[idx]}', c='#ffab00')
        ax[idx].plot(range(n_role_played_levels), results_cupa[idx]['avg_accuracy_per_model'], 'x:', label=f'CUPA - {labels[idx]}', c='#b83266')
        ax[idx].grid(alpha=0.5, axis='both')
        ax[idx].set_yticks(np.arange(0.4, 1.0, 0.1))
        ax[idx].set_xticks(range(n_role_played_levels))
        # ax[idx].set_xticklabels([x if x_idx%2 == 0 else '' for x_idx, x in enumerate(results_race[idx]['student_levels'])])
        ax[idx].set_xticklabels(results_race[idx]['student_levels'])
        # ax[idx].set_ylim(0.38, 0.92)
        ax[idx].legend()
        ax[idx].set_xlabel('Simulated level')
    ax[0].set_ylabel('MCQA accuracy')
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_44_45_46_gpt_3_5_race_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- additional analysis on language proficiency scales -- RACE and CUPA
    fig, ax = plt.subplots(3, 1, figsize=(6, 9))

    n_role_played_levels_0 = len(dict_gpt_3_5_race_60_test['student_levels'])
    ax[0].plot(range(n_role_played_levels_0), dict_gpt_3_5_race_60_test['avg_accuracy_per_model'], 'o:', label=f'RACE - CEFR (b)', c='#ffab00')
    ax[0].plot(range(n_role_played_levels_0), dict_gpt_3_5_cupa_60_test['avg_accuracy_per_model'], 'x:', label=f'CUPA - CEFR (b)', c='#b83266')

    n_role_played_levels_1 = len(dict_gpt_3_5_race_47_test['student_levels'])
    ax[1].plot(range(n_role_played_levels_1), dict_gpt_3_5_race_47_test['avg_accuracy_per_model'], 'o:', label=f'RACE - IELTS (b)', c='#ffab00')
    ax[1].plot(range(n_role_played_levels_1), dict_gpt_3_5_cupa_47_test['avg_accuracy_per_model'], 'o-', label=f'RACE - IELTS (b)', c='#ffab00')
    ax[1].plot(range(n_role_played_levels_1), dict_gpt_3_5_race_50_test['avg_accuracy_per_model'], 'x:', label=f'CUPA - IELTS (c)', c='#b83266')
    ax[1].plot(range(n_role_played_levels_1), dict_gpt_3_5_cupa_50_test['avg_accuracy_per_model'], 'x-', label=f'CUPA - IELTS (c)', c='#b83266')

    n_role_played_levels_2 = len(dict_gpt_3_5_race_49_test['student_levels'])
    ax[2].plot(range(n_role_played_levels_2), dict_gpt_3_5_race_49_test['avg_accuracy_per_model'], 'o:', label=f'RACE - TOEFL (b)', c='#ffab00')
    ax[2].plot(range(n_role_played_levels_2), dict_gpt_3_5_cupa_49_test['avg_accuracy_per_model'], 'x:', label=f'CUPA - TOEFL (b)', c='#b83266')
    ax[2].set_xlabel('Simulated level')

    results_for_xticks = [dict_gpt_3_5_race_60_test, dict_gpt_3_5_race_47_test, dict_gpt_3_5_race_49_test]
    for idx in range(3):
        n_role_played_levels = len(results_for_xticks[idx]['student_levels'])
        ax[idx].grid(alpha=0.5, axis='both')
        ax[idx].set_yticks(np.arange(0.4, 1.0, 0.1))
        ax[idx].set_ylabel('MCQA accuracy')
        ax[idx].set_xticks(range(n_role_played_levels))
        ax[idx].set_xticklabels(results_for_xticks[idx]['student_levels'])
        ax[idx].legend()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_47_49_50_60_gpt_3_5_race_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- reference prompts, GPT_3_5 vs. GPT_3_5_1106 -- RACE, CUPA, ARC
    n_role_played_levels = len(dict_gpt_3_5_arc_56_test['student_levels'])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_56_test['avg_accuracy_per_model'], '*-', label='ARC', c='#054b7d')
    ax.grid(alpha=0.5, axis='both')
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels([x if x_idx % 2 == 0 else '' for x_idx, x in enumerate(dict_gpt_3_5_arc_56_test['student_levels'])])
    ax.set_ylim(0.38, 0.92)
    ax.legend()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_56_gpt_3_5_arc_1106_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- exam grades -- ARC, RACE, and CUPA
    n_role_played_levels = len(dict_gpt_3_5_race_57_test['student_levels'])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_57_test['avg_accuracy_per_model'], 'o-', label='RACE', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_55_test['avg_accuracy_per_model'], '*-', label='ARC', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_cupa_57_test['avg_accuracy_per_model'], 'x-', label='CUPA', c='#b83266')
    ax.grid(alpha=0.5, axis='both')
    # ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_race_57_test['student_levels'])
    ax.legend()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_57_race_cupa_55_arc_gpt_3_5_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: average accuracy per model -- qualitative scale (beg, int, adv) -- ARC, RACE, and CUPA
    n_role_played_levels = len(dict_gpt_3_5_race_53_test['student_levels'])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_race_53_test['avg_accuracy_per_model'], 'o-', label='RACE', c='#ffab00')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_arc_52_test['avg_accuracy_per_model'], '*-', label='ARC', c='#054b7d')
    ax.plot(range(n_role_played_levels), dict_gpt_3_5_cupa_53_test['avg_accuracy_per_model'], 'x-', label='CUPA', c='#b83266')
    ax.grid(alpha=0.5, axis='both')
    # ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.set_ylabel('MCQA accuracy')
    ax.set_xlabel('Simulated level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_race_53_test['student_levels'])
    ax.legend()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_53_race_cupa_52_arc_gpt_3_5_mcqa_accuracy_per_level.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level comparison of different prompts -- ARC
    plot_style = ['*:', '*-.', '*--', '*-', '-']
    labels = [28, 31, 32, 35, 'ref. prompt']
    list_dict_results = [dict_gpt_3_5_arc_28_dev, dict_gpt_3_5_arc_31_dev, dict_gpt_3_5_arc_32_dev, dict_gpt_3_5_arc_35_dev, dict_gpt_3_5_arc_48_dev]
    n_role_played_levels = len(dict_gpt_3_5_arc_48_dev['student_levels'])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for idx, dict_results in enumerate(list_dict_results):
        color = '#288cfc' if idx == 4 else '#054b7d'
        ax.plot(range(n_role_played_levels), dict_results['avg_accuracy_per_model'], plot_style[idx], label=labels[idx], color=color)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.grid(alpha=0.5)
    ax.set_xlabel('Simulated level')
    ax.set_ylabel('MCQA accuracy')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_arc_48_dev['student_levels'])
    ax.legend()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_28_31_32_35_48_arc_gpt_3_5_mcqa_accuracy.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # FIGURE: MCQA by role played level comparison of different prompts (second version) -- ARC
    plot_style = ['*:', '*-.', '*--', '*-', '-']
    # labels = [28, 31, 32, 35, 'R.P.']
    labels = ['P.1', 'P.2', 'P.3', 'P.4', 'R.P.']
    list_dict_results = [dict_gpt_3_5_arc_28_dev, dict_gpt_3_5_arc_31_dev, dict_gpt_3_5_arc_32_dev, dict_gpt_3_5_arc_35_dev, dict_gpt_3_5_arc_48_dev]
    n_role_played_levels = len(dict_gpt_3_5_arc_48_dev['student_levels'])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for idx, dict_results in enumerate(list_dict_results):
        # color = '#288cfc' if idx == 4 else '#054b7d'
        color = '#054b7d' if idx == 4 else '#288cfc'
        ax.plot(range(n_role_played_levels), dict_results['avg_accuracy_per_model'], plot_style[idx], label=labels[idx], color=color)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.grid(alpha=0.5)
    ax.set_xlabel('Simulated level')
    ax.set_ylabel('MCQA accuracy')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(dict_gpt_3_5_arc_48_dev['student_levels'])
    ax.legend()
    plt.tight_layout()
    if DO_PLOT: plt.show()
    if SAVE_FIG: plt.savefig(os.path.join(out_fig_path, f'prompt_28_31_32_35_48_arc_gpt_3_5_mcqa_accuracy.pdf'))
    plt.close(fig)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":
    main()

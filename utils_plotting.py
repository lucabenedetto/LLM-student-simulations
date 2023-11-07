import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

COLORS = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
    'tab:purple', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:gray', 'tab:brown',
]


def plot_accuracy_per_model(
        average_accuracy_per_model,
        role_played_levels,
        dataset_name,
        prompt_idx,
        output_filepath: str = None,
):
    n_role_played_levels = len(role_played_levels)
    fig, ax = plt.subplots()
    ax.bar(range(n_role_played_levels), average_accuracy_per_model)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    ax.grid(alpha=0.5, axis='y')
    ax.set_ylabel('QA accuracy')
    ax.set_xlabel('Role-played level')
    ax.set_xticks(range(n_role_played_levels))
    ax.set_xticklabels(role_played_levels)
    ax.set_title(f'QA accuracy per role-played level | {dataset_name} | prompt {prompt_idx}')
    plt.show()


def plot_accuracy_per_difficulty_per_model(
        avg_accuracy_per_grade_per_model,
        dataset_name,
        prompt_idx,
        output_filepath: str = None,
):
    difficulty_levels = list(avg_accuracy_per_grade_per_model.keys())
    n_role_played_levels = len(avg_accuracy_per_grade_per_model[difficulty_levels[0]])
    for difficulty in difficulty_levels[1:]:
        if len(avg_accuracy_per_grade_per_model[difficulty]) != n_role_played_levels:
            print("WARNING!")  # TODO add raise error instead of this
    fig, ax = plt.subplots(1, len(difficulty_levels), sharey='all')
    for idx, grade in enumerate(difficulty_levels):
        ax[idx].set_ylim(0, 1.0)
        ax[idx].set_yticks(np.arange(0.0, 1.0, 0.1))
        ax[idx].grid(alpha=0.5, axis='y')
        ax[idx].bar(range(n_role_played_levels), avg_accuracy_per_grade_per_model[grade], color=COLORS[idx])
        ax[idx].plot([0, n_role_played_levels-1], [np.mean(avg_accuracy_per_grade_per_model[grade])] * 2, c='k')
        ax[idx].set_title(f'{dataset_name} | prompt {prompt_idx}\n Diff. level {grade} ')
        ax[idx].set_xlabel('Role-played level')
        ax[idx].set_ylabel('QA accuracy')
    plt.show()


def plot_accuracy_per_difficulty_for_different_role_played_levels(
        avg_accuracy_per_grade_per_model,
        role_played_levels,
        dataset_name,
        prompt_idx,
        output_filepath: str = None,
):
    difficulty_levels = list(avg_accuracy_per_grade_per_model.keys())
    n_role_played_levels = len(role_played_levels)

    fig, ax = plt.subplots(1, n_role_played_levels, sharey='all')
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
    plt.show()


def plot_correlation_between_difficulty_and_qa_correctness(
        correctness_per_model,
        difficulty_dict,
        dataset_name,
        prompt_idx,
        output_filepath: str = None,
):
    difficulty_levels = set(difficulty_dict.values())
    X, Y = [], []
    for q_id in correctness_per_model.keys():
        X.append(difficulty_dict[q_id])
        Y.append(np.mean(correctness_per_model[q_id]))

    # Version 1 of the plot: hexbin
    fig, ax = plt.subplots()
    # ax.scatter(X, Y)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(difficulty_levels)-0.2, max(difficulty_levels)+0.2)
    ax.hexbin(X, Y, gridsize=10)
    ax.set_title(f'Correlation between QA accuracy and true difficulty | {dataset_name} | prompt {prompt_idx}')
    ax.set_xlabel('"True" difficulty')
    ax.set_ylabel('QA correctness')
    m, b = np.polyfit(X, Y, 1)
    if m and b:
        x0, x1 = min(X), max(X)
        ax.plot([x0, x1], [x0 * m + b, x1 * m + b], c='r')
    plt.show()
    plt.close(fig)

    # Version 2 of the plot: KDE
    fig, ax = plt.subplots()
    ax.set_title(f'Correlation between QA accuracy and true difficulty | {dataset_name} | prompt {prompt_idx}')
    seaborn.kdeplot(pd.DataFrame({'"True" difficulty': X, 'QA correctness': Y}), x='"True" difficulty', y='QA correctness', fill=True, levels=15)
    if m and b:
        x0, x1 = min(X), max(X)
        ax.plot([x0, x1], [x0 * m + b, x1 * m + b], c='r')
    plt.show()
    plt.close(fig)

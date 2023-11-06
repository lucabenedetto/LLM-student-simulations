import matplotlib.pyplot as plt
import numpy as np


COLORS = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
    'tab:purple', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:gray', 'tab:brown',
]


def plot_accuracy_per_model(average_accuracy_per_model):
    n_role_played_levels = len(average_accuracy_per_model)
    fig, ax = plt.subplots()
    colors = ['darkred'] * 5 + ['darkgreen'] * 5 + ['orange'] * 5  # TODO!
    ax.bar(range(n_role_played_levels), average_accuracy_per_model, color=colors)
    plt.show()


def plot_accuracy_per_difficulty_per_model(avg_accuracy_per_grade_per_model):
    difficulty_levels = list(avg_accuracy_per_grade_per_model.keys())
    n_role_played_levels = len(avg_accuracy_per_grade_per_model[difficulty_levels[0]])
    for difficulty in difficulty_levels[1:]:
        if len(avg_accuracy_per_grade_per_model[difficulty]) != n_role_played_levels:
            print("WARNING!")  # TODO add raise error instead of this
    fig, ax = plt.subplots(1, len(difficulty_levels), sharey='all')
    for idx, grade in enumerate(difficulty_levels):
        ax[idx].grid(alpha=0.5)
        ax[idx].bar(range(n_role_played_levels), avg_accuracy_per_grade_per_model[grade], color=COLORS[idx])
        ax[idx].plot([0, n_role_played_levels-1], [np.mean(avg_accuracy_per_grade_per_model[grade])] * 2, c='k')
        ax[idx].set_title(f'Grade: {grade}')
        ax[idx].set_xlabel('Role-played level')
        ax[idx].set_ylabel('QA accuracy')
    plt.show()

import ast
import json
import os
from collections import defaultdict
from typing import Union, Dict, List, Set, Tuple

import pandas as pd

from constants import RACE, ARC, INPUT_DATA_DIR


def get_average_accuracy_per_model(
        filepaths,
        set_q_ids,
        complete_df,
        difficulty_levels,
) -> Tuple[List[float], Dict[int, List[float]]]:
    # dict that maps from "true_difficulty" to list of qids
    questions_by_difficulty = get_questions_by_difficulty_dict(complete_df, difficulty_levels)
    correct_answer_dict = get_correct_answer_dict_from_df(complete_df)  # dict that maps from qid to correct answer

    avg_accuracy_per_grade_per_model = dict()
    for grade in difficulty_levels:
        avg_accuracy_per_grade_per_model[grade] = []
    avg_accuracy_per_model = []
    for idx, filepath in enumerate(filepaths):
        df = pd.read_csv(filepath)
        df = df[df['q_id'].isin(set_q_ids)]
        # df = df[~df['q_id'].isna()]
        # df = df[~df['q_id'].isnull()]
        df['correct'] = df.apply(lambda r: int(correct_answer_dict[r['q_id']]) == get_answer_index_as_int(r['answer']), axis=1)
        avg_accuracy_per_model.append(df['correct'].mean())
        for grade in difficulty_levels:
            avg_accuracy_per_grade_per_model[grade].append(
                df[df['q_id'].isin(questions_by_difficulty[grade])]['correct'].mean())
    return avg_accuracy_per_model, avg_accuracy_per_grade_per_model


def get_response_correctness_per_model(
        filepaths,
        set_q_ids,
        complete_df,
) -> Tuple[Dict[str, List[bool]], Dict[str, List[str]]]:
    correct_answer_dict = get_correct_answer_dict_from_df(complete_df)  # I am computing this twice, I should probably pass it to the method instead of computing it here.

    correctness_per_model = defaultdict(list)
    answers_per_model = defaultdict(list)

    for idx, filepath in enumerate(filepaths):
        df = pd.read_csv(filepath)
        df = df[df['q_id'].isin(set_q_ids)]
        # df = df[~df['q_id'].isna()]
        # df = df[~df['q_id'].isnull()]

        # df['correct'] = df.apply(lambda r: int(correct_answer_dict[r['q_id']]) == get_answer_index_as_int(r['answer']), axis=1)
        # df['correct_answer'] = df.apply(lambda r: correct_answer_dict[r['q_id']], axis=1)
        for q_id, answer in df[['q_id', 'answer']].values:
            correctness_per_model[q_id].append(get_answer_index_as_int(answer) == int(correct_answer_dict[q_id]))
            answers_per_model[q_id].append(answer)
    return correctness_per_model, answers_per_model


def get_answer_index_as_int(answer):
    try:
        answer_integer = int(ast.literal_eval(answer)['index'])
    except:
        answer_integer = -9  # TODO fix this, this is tmp
    return answer_integer


def get_correct_answer_dict_from_df(df) -> Dict[str, str]:
    return {q_id: correct_answer for q_id, correct_answer in df[['q_id', 'correct_answer']].values}


def get_difficulty_dict_from_df(df) -> Dict[str, int]:
    return {q_id: difficulty for q_id, difficulty in df[['q_id', 'difficulty']].values}


def get_questions_by_difficulty_dict(df: pd.DataFrame, difficulty_levels: List[int]) -> Dict[int, Set[str]]:
    questions_by_difficulty = dict()
    for diff in difficulty_levels:
        questions_by_difficulty[diff] = set()
    for q_id, diff in df[['q_id', 'difficulty']].values:
        questions_by_difficulty[diff].add(q_id)
    return questions_by_difficulty


def get_dataset(dataset_name: str, num_questions_per_difficulty_level: int = 50) -> pd.DataFrame:
    if dataset_name == RACE and num_questions_per_difficulty_level == 50:
        return pd.read_csv(os.path.join(INPUT_DATA_DIR, "race_pp_test_50q_per_diff.csv"))
    elif dataset_name == ARC and num_questions_per_difficulty_level == 50:
        return pd.read_csv(os.path.join(INPUT_DATA_DIR, "arc_test_50q_per_diff.csv"))
    else:
        raise NotImplementedError()


def get_original_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == RACE:
        return pd.read_csv(os.path.join(INPUT_DATA_DIR, 'race_pp_test.csv'))
    elif dataset_name == ARC:
        return pd.read_csv(os.path.join(INPUT_DATA_DIR, 'arc_test.csv'))
    else:
        raise NotImplementedError()


def get_questions_answered_by_all_roleplayed_levels(filepaths, complete_df):
    set_q_ids = set(complete_df['q_id'].unique())
    for idx, filepath in enumerate(filepaths):
        df = pd.read_csv(filepath)
        df = df[df['answer'] != "{'index': -9, 'text': 'None'}"]
        set_q_ids = set_q_ids.intersection(set(df['q_id'].unique()))
    return set_q_ids


def get_student_levels_from_prompt_idx(prompt_idx):
    # thw two standard approaches with numbers in chars, these (especially 5 levels) are the ones used the most.
    five_levels_char = ['one', 'two', 'three', 'four', 'five']
    ten_levels_char = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    # the two "standard" approaches with numbers in digit, either 5 or 10 levels.
    five_levels_int = [str(idx) for idx in range(5)]
    ten_levels_int = [str(idx) for idx in range(10)]
    # "categorical" levels
    cat_levels = ['a beginner', 'an intermediate', 'an expert']
    # grades marks
    mark_grades = ['A', 'B', 'C', 'D', 'F']
    # school_grades
    school_grades = ['a third grade', 'a fourth grade', 'a fifth grade', 'a sixth grade',
                     'a seventh grade', 'an eight grade', 'a ninth grade']
    # IELTS levels
    ielts_levels = ['4', '4.5', '5', '5.5', '6', '6.5', '7', '7.5', '8', '9']
    ielts_levels_2 = ['4', '5', '6', '7', '8', '9']
    ielts_levels_with_def = ['4 (Limited test taker)', '5 (Modest test taker)', '6 (Competent test taker)',
                             '7 (Good test taker)', '8 (Very good test taker)', '9 (Expert test taker)']
    ielts_definitions_without_numbers = ['Limited test taker', 'Modest test taker', 'Competent test taker',
                                         'Good test taker', 'Very good test taker', 'Expert test taker']
    # CEFR levels
    cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    # TOEFL levels
    toefl_levels = ['32', '35', '46', '60', '79', '94', '102', '110', '115', '118']
    rounded_toefl_levels = ['40', '60', '80', '100', '120']  # this was used to see if "rounded" nums work better

    if prompt_idx in {39, 40, 41, 48, 54}:
        return five_levels_char
    if prompt_idx in {43}:
        return five_levels_int
    if prompt_idx in {44}:
        return cefr_levels
    if prompt_idx in {45}:
        return ielts_levels
    if prompt_idx in {46}:
        return toefl_levels
    if prompt_idx in {47}:
        return ielts_levels_2
    if prompt_idx in {49}:
        return rounded_toefl_levels
    if prompt_idx in {50}:
        return ielts_levels_with_def
    if prompt_idx in {51}:
        return ielts_definitions_without_numbers
    if prompt_idx in {52, 53}:
        return cat_levels
    if prompt_idx in {55, 57}:
        return mark_grades
    if prompt_idx in {56}:
        return school_grades
    raise NotImplementedError()


def build_system_message_from_params(prompt_idx, student_level):
    if prompt_idx == 39:
        return f"""
You are taking a science exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of level {student_level}"}}
"""
    if prompt_idx == 40:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of level {student_level}"}}        
"""
    if prompt_idx == 41:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "index": "integer index of the answer chosen by a student of level {student_level}"}}        
"""
    if prompt_idx == 43:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from 1 (very easy) to 5 (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of level {student_level}"}}        
"""
    if prompt_idx == 44:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of CEFR level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of CEFR level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of CEFR level {student_level}"}}        
"""
    if prompt_idx == 45 or prompt_idx == 50:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of IELTS level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of IELTS level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of IELTS level {student_level}"}}    
"""
    if prompt_idx == 46 or prompt_idx == 49:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of TOEFL level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of TOEFL level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of TOEFL level {student_level}"}}        
"""
    if prompt_idx == 47:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of IELTS level {student_level} would pick.
The meaning of the IELTS levels is as follows:
- IELTS level 9 indicates an Expert test taker
- IELTS level 8 indicates a Very good test taker
- IELTS level 7 indicates a Good test taker
- IELTS level 6 indicates a Competent test taker
- IELTS level 5 indicates a Modest test taker
- IELTS level 4 indicates a Limited test taker
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of IELTS level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of IELTS level {student_level}"}}
"""
    if prompt_idx == 48:
        return f"""
You will be shown a multiple choice question from a science exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of level {student_level}"}}
"""
    if prompt_idx == 51:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the a {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a {student_level}"}}
"""
    if prompt_idx == 52:
        return f"""
You will be shown a multiple choice question from a science exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that {student_level} student would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that {student_level} student would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by {student_level} student"}}
"""
    if prompt_idx == 53:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that {student_level} student would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that {student_level} student would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by {student_level} student"}}
"""
    if prompt_idx == 54:
        return f"""
You are taking an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of level {student_level}"}}
"""
    if prompt_idx == 55:
        return f"""
You will be shown a multiple choice question from a science exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a grade {student_level} student would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that a grade {student_level} student would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a grade {student_level} student"}}
"""
    if prompt_idx == 56:
        return f"""
You will be shown a multiple choice question from a science exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that {student_level} student would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that {student_level} student would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by {student_level} student"}}
"""
    if prompt_idx == 57:
        return f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a grade {student_level} student would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that a grade {student_level} student would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a grade {student_level} student"}}
"""
    raise NotImplementedError()


def build_user_prompt_from_params(question, answers, is_reading_question, context=None) -> str:
    if is_reading_question:
        prompt = f"""
Reading passage: "{context}"
Question: "{question}"
Options: "{answers}"
"""
    else:
        prompt = f"""
Question: "{question}"
Options: "{answers}"
"""
    return prompt


def validate_answer(answer: str) -> Union[dict, None]:
    try:
        answer_json = json.loads(answer)
        index_str = str(answer_json['index'])
        if not index_str.isdigit():
            print("The index is not an integer.")
            return None
        return answer_json
    except json.JSONDecodeError:
        print("The answer is not a valid JSON string.")
        return None

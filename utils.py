import json
import os
from typing import Union

import pandas as pd

from constants import RACE, ARC, DATA_DIR


def get_dataset(dataset_name: str, num_questions_per_difficulty_level: int = 50) -> pd.DataFrame:
    if dataset_name == RACE and num_questions_per_difficulty_level == 50:
        return pd.read_csv(os.path.join(DATA_DIR, "processed/race_pp_test_50q_per_diff.csv"))
    elif dataset_name == ARC and num_questions_per_difficulty_level == 50:
        return pd.read_csv(os.path.join(DATA_DIR, "processed/arc_test_50q_per_diff.csv"))
    else:
        raise NotImplementedError()


def get_student_levels_from_prompt_idx(prompt_idx):
    # thw two standard approaches with numbers in chars, these (especially 5 levels) are the ones used the most.
    five_levels_char = ['one', 'two', 'three', 'four', 'five']
    ten_levels_char = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    # the two "standard" approaches with numbers in digit, either 5 or 10 levels.
    five_levels_int = [str(idx) for idx in range(6)]
    ten_levels_int = [str(idx) for idx in range(10)]
    # IELTS levels
    ielts_levels = ['4', '4.5', '5', '5.5', '6', '6.5', '7', '7.5', '8', '9']
    ielts_levels_2 = ['4', '5', '6', '7', '8', '9']
    # CEFR levels
    cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    # TOEFL levels
    toefl_levels = ['32', '35', '46', '60', '79', '94', '102', '110', '115', '118']
    rounded_toefl_levels = ['45', '60', '80', '95', '100', '120']  # this was used to see if "rounded" nums work better

    student_levels_list = None
    if prompt_idx in {40, 41}:
        student_levels_list = five_levels_char
    if prompt_idx in {43}:
        student_levels_list = five_levels_int
    if prompt_idx in {44}:
        student_levels_list = cefr_levels
    if prompt_idx in {45}:
        student_levels_list = ielts_levels
    if prompt_idx in {46}:
        student_levels_list = toefl_levels
    elif prompt_idx in {47}:
        student_levels_list = ielts_levels_2
    else:
        raise NotImplementedError()
    return student_levels_list


def build_system_message_from_params(prompt_idx, student_level):
    system_message = ''
    if prompt_idx == 39:
        system_message = f"""
You are taking a science exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of level {student_level}"}}
"""
    if prompt_idx == 40:
        system_message = f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of level {student_level}"}}        
"""
    if prompt_idx == 41:
        system_message = f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "index": "integer index of the answer chosen by a student of level {student_level}"}}        
"""
    if prompt_idx == 43:
        system_message = f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from 1 (very easy) to 5 (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of level {student_level}"}}        
"""
    if prompt_idx == 44:
        system_message = f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of CEFR level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of CEFR level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of CEFR level {student_level}"}}        
"""
    if prompt_idx == 45:
        system_message = f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of IELTS level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of IELTS level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of IELTS level {student_level}"}}    
"""
    if prompt_idx == 46:
        system_message = f"""
You will be shown a multiple choice question from an English reading comprehension exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of TOEFL level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of TOEFL level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of TOEFL level {student_level}"}}        
"""
    elif prompt_idx == 47:
        system_message = f"""
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
    else:
        raise NotImplementedError()
    return system_message


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

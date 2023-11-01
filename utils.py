def get_student_levels_from_prompt_idx(prompt_idx):
    # the two "standard" approaches, either 5 or 10 levels.
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

    if prompt_idx in {47}:
        student_levels_list = ielts_levels_2
    else:
        raise NotImplementedError()
    return student_levels_list


def _build_system_message_from_params(prompt_idx, student_level):
    if prompt_idx == 47:
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

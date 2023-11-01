from typing import Union
import json
import openai

from utils import _build_system_message_from_params


def answer_question(system_context: str, user_prompt: str, temperature=0, model=None) -> str:
    if model is None:
        raise ValueError
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'system', 'content': system_context},
                  {'role': 'user', 'content': user_prompt}],
        temperature=temperature,
    )
    response_content = response['choices'][0]['message']['content']
    return response_content


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


def build_system_message_from_params(prompt_idx, student_level) -> str:
    return _build_system_message_from_params(prompt_idx, student_level)


def prepare_answers_dict(df_questions, model, student_level=None, is_reading_question=False, prompt_idx=None):
    answers_dict = {}
    for idx, row in df_questions.iterrows():
        print("Processing idx: ", idx)
        prompt = build_user_prompt_from_params(row.question, row.options, is_reading_question, row.context)
        system_message = build_system_message_from_params(prompt_idx, student_level)
        try:
            answer = answer_question(system_message, prompt, model=model)
            answer = validate_answer(answer)
        except Exception as e:
            print(e)
            answer = "{'index': -9, 'text': 'None'}"  # this if the GPT model did not produce a valid JSON or integer
        answers_dict[row.q_id] = answer
    return answers_dict


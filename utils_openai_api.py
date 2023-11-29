import openai
import pandas as pd

from utils import build_system_message_from_params, build_user_prompt_from_params, validate_answer
from constants import (
    GPT_3_5,
    GPT_3_5_1106,
)


def get_gpt_model(name):
    if name == GPT_3_5:
        return 'gpt-3.5-turbo-0613'
    elif name == GPT_3_5_1106:
        return 'gpt-3.5-turbo-1106'
    else:
        raise ValueError("Unknown model")


def answer_question(system_context: str, user_prompt: str, temperature=0, model=None, response_format='text') -> str:
    if model is None:
        raise ValueError
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'system', 'content': system_context},
                  {'role': 'user', 'content': user_prompt}],
        temperature=temperature,
        response_format={"type": response_format},
    )
    response_content = response['choices'][0]['message']['content']
    return response_content


def build_gpt_system_message_from_params(prompt_idx, student_level) -> str:
    return build_system_message_from_params(prompt_idx, student_level)


def prepare_answers_dict_gpt(
        df_questions: pd.DataFrame,
        model: str,
        student_level: int = None,
        is_reading_question: bool = False,
        prompt_idx: int = None,
        response_format: str = 'text',
):
    answers_dict = {}
    for idx, row in df_questions.iterrows():
        print("Processing idx: ", idx)
        prompt = build_user_prompt_from_params(row.question, row.options, is_reading_question, row.context)
        system_message = build_gpt_system_message_from_params(prompt_idx, student_level)
        try:
            answer = answer_question(system_message, prompt, model=model, response_format=response_format)
            # answer = validate_answer(answer)
        except Exception as e:
            print(e)
            answer = "{'index': -9, 'text': 'None'}"  # this if the GPT model did not produce a response
        answers_dict[row.q_id] = answer
    return answers_dict


import openai

from utils import build_system_message_from_params, build_user_prompt_from_params, validate_answer
from constants import (
    GPT_3_5,
    GPT_3_5_1106,
    GPT_4_1106
)


def get_gpt_model(name):
    if name == GPT_3_5:
        return 'gpt-3.5-turbo-0613'
    elif name == GPT_3_5_1106:
        return 'gpt-3.5-turbo-1106'
    elif name == GPT_4_1106:
        return 'gpt-4-1106-preview'
    else:
        raise ValueError("Unknown model")


def answer_question(system_context: str, user_prompt: str, temperature=0, model=None, json_mode=True) -> str:
    if model is None:
        raise ValueError
    type_format = 'json_object' if json_mode and model == 'gpt-4-1106-preview' else 'text'
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'system', 'content': system_context},
                  {'role': 'user', 'content': user_prompt}],
        temperature=temperature,
        response_format={"type": type_format}
    )
    response_content = response['choices'][0]['message']['content']
    return response_content


def build_gpt_system_message_from_params(prompt_idx, student_level) -> str:
    return build_system_message_from_params(prompt_idx, student_level)


def prepare_answers_dict_gpt(df_questions, model, student_level=None, is_reading_question=False, prompt_idx=None):
    answers_dict = {}
    for idx, row in df_questions.iterrows():
        print("Processing idx: ", idx)
        prompt = build_user_prompt_from_params(row.question, row.options, is_reading_question, row.context)
        system_message = build_gpt_system_message_from_params(prompt_idx, student_level)
        try:
            answer = answer_question(system_message, prompt, model=model)
            # answer = validate_answer(answer)
        except Exception as e:
            print(e)
            answer = "{'index': -9, 'text': 'None'}"  # this if the GPT model did not produce a response
        answers_dict[row.q_id] = answer
    return answers_dict


import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import transformers
import torch

from utils import (
    get_student_levels_from_prompt_idx,
    get_dataset,
    build_user_prompt_from_params,
    build_system_message_from_params,
    validate_answer,
)
from constants import RACE, ARC, IS_READING_QUESTION, OUTPUT_DATA_DIR


def clean_raw_llama_answer(answer):
    answer = answer.split('{')[1]
    answer = answer.split('}')[0]
    answer = '{' + answer + '}'
    answer = validate_answer(answer)
    return answer


def get_llama_input_prompt(student_level, prompt_idx, is_reading_question, question, options, context):
    return f"""[INST] <<SYS>> \
{build_system_message_from_params(prompt_idx, student_level)} <</SYS>>
{build_user_prompt_from_params(question, options, is_reading_question, context)} [/INST]"""


def prepare_answers_dict_llama(df_questions, pipeline, student_level=None, is_reading_question=False, prompt_idx=None):
    answers_dict = {}

    df_questions['input_prompt'] = df_questions.apply(
        lambda r: get_llama_input_prompt(student_level, prompt_idx, is_reading_question, r['question'], r['options'], r['context']),
        axis=1
    )
    list_q_id = df_questions['q_id'].values.tolist()
    print(list_q_id)

    sequences = pipeline(
        df_questions['input_prompt'].values.tolist(),  # I call the pipeline on the whole dataset. because it is much more efficient.
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        return_full_text=False,
        eos_token_id=None,  # tokenizer.eos_token_id,
        max_length=750, # this is important to get right especially for the reading comprehension questions, as they can be quite long.
    )
    # len of sequences is the number of elements in df_questions.
    for idx, answer in enumerate(sequences):
        try:
            # answer is a list, with one element only
            answer = answer[0]['generated_text']
        except Exception as e:
            print(e)
            answer = "{'index': -9, 'text': 'None'}"  # this if the model did not produce a valid JSON or integer
        answers_dict[list_q_id[idx]] = answer
    return answers_dict


DATASET = ARC
# folder_name = '23_11_llama_responses_race_pp'  # RACE
folder_name = '23_11_llama_responses_arc'  # ARC

PROMPT_IDX = 39

st_levels = get_student_levels_from_prompt_idx(PROMPT_IDX)
df_items = get_dataset(DATASET, 50)

df_items = df_items[:11]  # todo this is tmp

is_reading_question = IS_READING_QUESTION[DATASET]

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

for idx, student_level in enumerate(st_levels):
    print(f"-- Doing idx {idx}, student level {student_level}")
    answers_dict = prepare_answers_dict_llama(df_items, pipeline, student_level, is_reading_question=is_reading_question, prompt_idx=PROMPT_IDX)

    rows = []
    for item, value in answers_dict.items():
        row = []
        row.append(item)
        row.append(value)
        rows.append(row)

    df_model_answers = pd.DataFrame(rows, columns=['q_id', 'raw_answer'])
    df_model_answers['answer'] = df_model_answers.apply(lambda r: clean_raw_llama_answer(r['raw_answer']), axis=1)

    df_model_answers.to_csv(
        os.path.join(OUTPUT_DATA_DIR, folder_name, f"llama2_answers_prompt{PROMPT_IDX}_0shot_{student_level}.csv"),
        index=False
    )

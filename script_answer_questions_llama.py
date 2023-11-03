import os
import pandas as pd
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
from constants import RACE, ARC, DATA_DIR, IS_READING_QUESTION, OUTPUT_DIR


def prepare_answers_dict_llama(df_questions, pipeline, student_level=None, is_reading_question=False, prompt_idx=None):
    answers_dict = {}
    for idx, row in df_questions.iterrows():
        print("Processing idx: ", idx)
        prompt = build_user_prompt_from_params(row.question, row.options, is_reading_question, row.context)
        system_message = build_system_message_from_params(prompt_idx, student_level)
        input_prompt = f"""<s>[INST] <<SYS>>
        {system_message}
        <</SYS>>

        {prompt} [/INST]"""
        try:
            sequences = pipeline(
                input_prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                return_full_text=False,
                eos_token_id=tokenizer.eos_token_id,
                max_length=750,  # this is important to get right especially for the reading comprehension questions, as they can be quite long.
            )
            answer = sequences[0]
            answer = validate_answer(answer)
        except Exception as e:
            print(e)
            answer = "{'index': -9, 'text': 'None'}"  # this if the model did not produce a valid JSON or integer
        answers_dict[row.q_id] = answer
    return answers_dict


DATASET = ARC
# folder_name = '23_11_llama_responses_race_pp'  # RACE
folder_name = '23_11_llama_responses'  # ARC

PROMPT_IDX = 47

st_levels = get_student_levels_from_prompt_idx(PROMPT_IDX)
df_items = get_dataset(DATASET, 50)

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

    df_model_answers = pd.DataFrame(rows, columns=["q_id", "answer"])

    df_model_answers.to_csv(
        os.path.join(DATA_DIR, OUTPUT_DIR, folder_name, f"llama2_answers_prompt{PROMPT_IDX}_0shot_{student_level}.csv"),
        index=False
    )

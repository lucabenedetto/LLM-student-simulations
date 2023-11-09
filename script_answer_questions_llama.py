import os
import pandas as pd
from transformers import AutoTokenizer
import transformers
import torch

from utils import (
    get_student_levels_from_prompt_idx,
    get_dataset,
)
from constants import ARC, IS_READING_QUESTION, OUTPUT_DATA_DIR
from utils_llama import clean_raw_llama_answer, prepare_answers_dict_llama

DATASET = ARC
# folder_name = '23_11_llama_responses_race_pp'  # RACE
folder_name = '23_11_llama_responses_arc'  # ARC

PROMPT_IDX = 39

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

    df_model_answers = pd.DataFrame(rows, columns=['q_id', 'raw_answer'])
    df_model_answers['answer'] = df_model_answers.apply(lambda r: clean_raw_llama_answer(r['raw_answer']), axis=1)

    # the 1+idx is needed for backward compatibility with files written with a previous script.
    df_model_answers.to_csv(
        os.path.join(OUTPUT_DATA_DIR, folder_name, f"llama2_answers_prompt{PROMPT_IDX}_0shot_{1+idx}.csv"),
        index=False
    )

print(f"[INFO] Complete run Llama | {DATASET} | prompt {PROMPT_IDX}")

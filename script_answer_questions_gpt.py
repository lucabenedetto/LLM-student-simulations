import json
import os
import pandas as pd
import openai

from utils import get_student_levels_from_prompt_idx, get_dataset
from utils_openai_api import prepare_answers_dict_gpt
from constants import RACE, ARC, IS_READING_QUESTION, OUTPUT_DATA_DIR

# get the OpenAI API key
with open('key.json', 'r') as f:
    data = json.load(f)
api_key = data['key']
openai.api_key = api_key

model = 'gpt-3.5-turbo'

DATASET = RACE
PROMPT_IDX = 47

st_levels = get_student_levels_from_prompt_idx(PROMPT_IDX)
df_items = get_dataset(DATASET, 50)

is_reading_question = IS_READING_QUESTION[DATASET]

folder_name = f'gpt_responses_{DATASET}'

for idx, student_level in enumerate(st_levels):
    print(f"-- Doing idx {idx}, student level {student_level}")
    answers_dict = prepare_answers_dict_gpt(df_items, model, student_level, is_reading_question=is_reading_question, prompt_idx=PROMPT_IDX)

    rows = []
    for item, value in answers_dict.items():
        row = []
        row.append(item)
        row.append(value)
        rows.append(row)

    df_model_answers = pd.DataFrame(rows, columns=["q_id", "answer"])

    # the 1+idx is needed for backward compatibility with files written with a previous script.
    df_model_answers.to_csv(
        os.path.join(OUTPUT_DATA_DIR, folder_name, f"gpt3_5_grade_answers_prompt{PROMPT_IDX}_0shot_a_{1+idx}.csv"),
        index=False
    )

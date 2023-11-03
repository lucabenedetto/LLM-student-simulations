import json
import os
import pandas as pd
import openai

from utils import get_student_levels_from_prompt_idx, get_dataset
from utils_openai_api import prepare_answers_dict_gpt
from constants import RACE, ARC, DATA_DIR, IS_READING_QUESTION, OUTPUT_DIR

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

# folder_name = '2309_gpt_responses_race_pp'  # RACE
folder_name = '23_07_gpt_responses'  # ARC

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

    df_model_answers.to_csv(
        os.path.join(DATA_DIR, OUTPUT_DIR, folder_name, f"gpt3_5_grade_answers_prompt{PROMPT_IDX}_0shot_a_{student_level}.csv"),
        index=False
    )

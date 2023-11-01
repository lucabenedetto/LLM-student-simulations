import json
import os
import pandas as pd
import openai

from utils import get_student_levels_from_prompt_idx
from utils_openai_api import prepare_answers_dict
from constants import RACE, ARC, DATA_DIR

# get the OpenAI API key
with open('key.json', 'r') as f:
    data = json.load(f)
api_key = data['key']
openai.api_key = api_key

model = 'gpt-3.5-turbo'

DATASET = RACE
PROMPT_IDX = 47

st_levels = get_student_levels_from_prompt_idx(PROMPT_IDX)

if DATASET == RACE:
    df_items = pd.read_csv(os.path.join(DATA_DIR, "processed/race_pp_test_50q_per_diff.csv"))
    out_dir = 'output/2309_gpt_responses_race_pp'
    is_reading_question = True
elif DATASET == ARC:
    df_items = pd.read_csv(os.path.join(DATA_DIR, "processed/arc_test_50q_per_diff.csv"))
    out_dir = 'output/23_07_gpt_responses'
    is_reading_question = False
else:
    raise NotImplementedError

for idx, student_level in enumerate(st_levels):
    print(f"-- Doing idx {idx}, student level {student_level}")
    answers_dict = prepare_answers_dict(df_items, model, student_level, is_reading_question=is_reading_question, prompt_idx=PROMPT_IDX)

    rows = []
    for item, value in answers_dict.items():
        row = []
        row.append(item)
        row.append(value)
        rows.append(row)

    df_model_answers = pd.DataFrame(rows, columns=["q_id", "answer"])

    df_model_answers.to_csv(
        os.path.join(DATA_DIR, out_dir, f"gpt3_5_grade_answers_prompt{PROMPT_IDX}_0shot_a_{student_level}.csv"),
        index=False
    )

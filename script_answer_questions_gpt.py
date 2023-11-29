import json
import os
import pandas as pd
import openai

from utils import get_student_levels_from_prompt_idx, get_dataset
from utils_openai_api import prepare_answers_dict_gpt, get_gpt_model, validate_answer
from constants import (
    RACE,
    ARC,
    IS_READING_QUESTION,
    OUTPUT_DATA_DIR,
    GPT_3_5,
    GPT_3_5_1106,
    GPT_4_1106
)

DATASET = RACE
PROMPT_IDX = 59
MODEL = GPT_3_5


def main():
    # get the OpenAI API key
    with open('../unsupervised-qdet/key.json', 'r') as f:
        data = json.load(f)
    api_key = data['key']
    openai.api_key = api_key

    st_levels = get_student_levels_from_prompt_idx(PROMPT_IDX)
    df_items = get_dataset(DATASET, 50)
    is_reading_question = IS_READING_QUESTION[DATASET]
    folder_name = f'{MODEL}_responses_{DATASET}'
    if not os.path.exists(os.path.join(OUTPUT_DATA_DIR, folder_name)):
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, folder_name))
    model = get_gpt_model(MODEL)

    for idx, student_level in enumerate(st_levels):
        print(f"-- Doing idx {idx}, student level {student_level}")
        answers_dict = prepare_answers_dict_gpt(df_items, model, student_level, is_reading_question=is_reading_question, prompt_idx=PROMPT_IDX)

        df_model_answers = pd.DataFrame([(item, value) for item, value in answers_dict.items()], columns=['q_id', 'raw_answer'])
        df_model_answers['answer'] = df_model_answers.apply(lambda r: validate_answer(r['raw_answer']), axis=1)

        # the 1+idx is needed for backward compatibility with files written with a previous script.
        df_model_answers.to_csv(
            os.path.join(OUTPUT_DATA_DIR, folder_name, f"{MODEL}_grade_answers_prompt{PROMPT_IDX}_0shot_a_{1+idx}.csv"),
            index=False
        )

    print(f"[INFO] Complete run {MODEL} | {DATASET} | prompt {PROMPT_IDX}")


if __name__ == "__main__":
    main()

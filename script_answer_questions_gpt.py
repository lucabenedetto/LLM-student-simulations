import json
import os
import pandas as pd
import openai

from utils import get_student_levels_from_prompt_idx, get_dataset
from utils_openai_api import prepare_answers_dict_gpt
from constants import RACE, ARC, IS_READING_QUESTION, OUTPUT_DATA_DIR, GPT_3_5

DATASET = RACE
PROMPT_IDX = 47
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
    # todo: change this depending on param so that it will work with GPT-4 as well
    model = 'gpt-3.5-turbo-0613'

    for idx, student_level in enumerate(st_levels):
        print(f"-- Doing idx {idx}, student level {student_level}")
        answers_dict = prepare_answers_dict_gpt(df_items, model, student_level, is_reading_question=is_reading_question, prompt_idx=PROMPT_IDX)

        rows = []
        for item, value in answers_dict.items():
            row = []
            row.append(item)
            row.append(value)
            rows.append(row)

        # TODO: as of now, for GPT I don't save the raw answer but only the processed one. I might want to change this.
        #   see below for the corresponding code from the llama runs.
        df_model_answers = pd.DataFrame(rows, columns=["q_id", "answer"])
        # df_model_answers = pd.DataFrame(rows, columns=['q_id', 'raw_answer'])
        # df_model_answers['answer'] = df_model_answers.apply(lambda r: clean_raw_gpt_answer(r['raw_answer']), axis=1)

        # the 1+idx is needed for backward compatibility with files written with a previous script.
        df_model_answers.to_csv(
            os.path.join(OUTPUT_DATA_DIR, folder_name, f"{MODEL}_grade_answers_prompt{PROMPT_IDX}_0shot_a_{1+idx}.csv"),
            index=False
        )

    print(f"[INFO] Complete run {MODEL} | {DATASET} | prompt {PROMPT_IDX}")


if __name__ == "__main__":
    main()

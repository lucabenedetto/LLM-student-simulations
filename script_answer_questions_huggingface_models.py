import os
import pandas as pd
from transformers import AutoTokenizer
import transformers
import torch

from utils import (
    get_student_levels_from_prompt_idx,
    get_dataset,
)
from constants import (
    ARC,
    RACE,
    IS_READING_QUESTION,
    OUTPUT_DATA_DIR,
    LLAMA2_13B_CHAT,
    LLAMA2_7B_CHAT,
    VICUNA_13B_V1_5,
)
from utils_huggingface_models import clean_raw_llama_answer, prepare_answers_dict_huggingface_model, get_llama_model

DATASET = RACE
PROMPT_IDX = 40
MAX_LENGTH = 2048
MODEL = VICUNA_13B_V1_5
SPLIT = 'test'


def main():

    folder_name = f'{MODEL}_responses_{DATASET}'
    st_levels = get_student_levels_from_prompt_idx(PROMPT_IDX)
    df_items = get_dataset(DATASET, n_questions_per_diff_level=50, split=SPLIT)
    is_reading_question = IS_READING_QUESTION[DATASET]
    model = get_llama_model(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,  # This is tmp, to check
        max_length=MAX_LENGTH,  # This is tmp, to check
        temperature=0.7,  # This is tmp, to check
        top_p=1,  # This is tmp, to check
        # repetition_penalty=1.15,  # This is tmp, to check
        # torch_dtype=torch.float16,  # This was here but removed for the time being.
        # device_map="auto",  # This was here but removed for the time being.
    )

    for idx, student_level in enumerate(st_levels):
        print(f"-- Doing idx {idx}, student level {student_level}")
        answers_dict = prepare_answers_dict_huggingface_model(
            MODEL,
            df_items,
            pipeline,
            student_level,
            is_reading_question=is_reading_question,
            prompt_idx=PROMPT_IDX,
            max_length=MAX_LENGTH,
        )

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
            os.path.join(OUTPUT_DATA_DIR, SPLIT, folder_name, f"{MODEL}_answers_prompt{PROMPT_IDX}_0shot_{1+idx}.csv"),
            index=False
        )

    print(f"[INFO] Complete run {MODEL} | {DATASET} | prompt {PROMPT_IDX}")


if __name__ == "__main__":
    main()

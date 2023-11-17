RACE = 'race_pp'
ARC = 'arc'
CUPA = 'cupa'

GPT_3_5 = 'gpt3_5'
GPT_3_5_1106 = 'gpt3_5_1106'
LLAMA2_7B_CHAT = 'llama2_7b_chat'
LLAMA2_13B_CHAT = 'llama2_13b_chat'
LLAMA2_13B = 'llama2_13b'

IS_READING_QUESTION = {
    RACE: True,
    ARC: False,
}

DIFFICULTY_LEVELS = {
    RACE: [0, 1, 2],
    ARC: [3, 4, 5, 6, 7, 8, 9],
    CUPA: [],
}

INPUT_DATA_DIR = 'data/input'
OUTPUT_DATA_DIR = 'data/output'

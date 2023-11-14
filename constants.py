RACE = 'race_pp'
ARC = 'arc'

LLAMA2_7B = 'llama2_7b'
LLAMA2_13B = 'llama2_13b'

IS_READING_QUESTION = {
    RACE: True,
    ARC: False,
}

DIFFICULTY_LEVELS = {
    RACE: [0, 1, 2],
    ARC: [3, 4, 5, 6, 7, 8, 9],
}

INPUT_DATA_DIR = 'data/input'
OUTPUT_DATA_DIR = 'data/output'

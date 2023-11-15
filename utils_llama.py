from utils import validate_answer, build_system_message_from_params, build_user_prompt_from_params
from constants import LLAMA2_13B, LLAMA2_7B


def get_index_from_raw_answer(answer):
    try:
        answer = answer.lower()
        answer = answer.split('{')[1]
        answer = answer.split('}')[0]
        answer = '{' + answer + '}'
        answer = answer.replace('\n', ' ')
        answer = answer.replace('  ', ' ')
        answer = answer.replace('"', '')
        answer = answer.split('index: ')[1][0]
        answer = int(answer)
    except:
        # print(answer)
        answer = -9
    return answer


def get_model(name):
    if name == LLAMA2_7B:
        return "meta-llama/Llama-2-7b-chat-hf"
    elif name == LLAMA2_13B:
        return "meta-llama/Llama-2-13b-chat-hf"
    else:
        raise ValueError("Unknown model")


def clean_raw_llama_answer(answer):
    try:
        answer = answer.split('{')[1]
        answer = answer.split('}')[0]
        answer = '{' + answer + '}'
        answer = validate_answer(answer)
    except Exception as e:
        print(e)
        answer = "{'index': -9, 'text': 'None'}"  # this if the model did not produce a valid JSON or integer
    return answer


def get_llama_input_prompt(student_level, prompt_idx, is_reading_question, question, options, context):
    return f"""[INST] <<SYS>> \
{build_system_message_from_params(prompt_idx, student_level)} <</SYS>>
{build_user_prompt_from_params(question, options, is_reading_question, context)} [/INST]"""


def prepare_answers_dict_llama(
        df_questions,
        pipeline,
        student_level=None,
        is_reading_question=False,
        prompt_idx=None,
        num_return_sequences=1,
        return_full_text=False,
        eos_token_id=None,
        max_length=750,
):
    answers_dict = {}

    df_questions['input_prompt'] = df_questions.apply(
        lambda r: get_llama_input_prompt(student_level, prompt_idx, is_reading_question, r['question'], r['options'], r['context']),
        axis=1
    )
    list_q_id = df_questions['q_id'].values.tolist()

    sequences = pipeline(
        df_questions['input_prompt'].values.tolist(),
        do_sample=True,
        top_k=10,
        num_return_sequences=num_return_sequences,
        return_full_text=return_full_text,
        eos_token_id=eos_token_id,  # tokenizer.eos_token_id,
        max_length=max_length,
    )
    # len of sequences is the number of elements in df_questions.
    for idx, answer in enumerate(sequences):
        try:
            # answer is a list, with one element only
            answer = answer[0]['generated_text']
        except Exception as e:
            print(e)
            answer = "{'index': -9, 'text': 'None'}"  # this if the model did not produce a valid JSON or integer
        answers_dict[list_q_id[idx]] = answer
    return answers_dict

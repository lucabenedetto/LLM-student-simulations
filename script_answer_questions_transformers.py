from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

question = 'Screech owls nest in holes in trees. They eat mice and crickets. Where would a screech owl most likely live?'
answers = "['a beach ', 'a forest ', 'a desert ', 'a rainforest']"
student_level = 1
system_message = f"""
You will be shown a multiple choice question from an science exam, and the questions in the exam have difficulty levels on a scale from one (very easy) to five (very difficult).
You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level} would pick.
Provide only a JSON file with the following structure:
{{"question level": "difficulty level of the question", "answer explanation": "the list of steps that the students of level {student_level} would follow to select the answer, including the misconceptions that might cause them to make mistakes", "index": "integer index of the answer chosen by a student of level {student_level}"}}        
"""
user_prompt = f"""
Question: "{question}"
Options: "{answers}"
"""


# input_prompt = system_message + '\n' + user_prompt
input_prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{user_prompt} [/INST]
"""
sequences = pipeline(
    input_prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=750,  # this is important to get right especially for the reading comprehension questions, as they can be quite long.
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
    print("----")
    print(len(seq))
    print("----")
    print(len(input_prompt))

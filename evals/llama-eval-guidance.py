#!/usr/bin/env python3

import os

import pandas as pd
import spaces
import torch
from tqdm import tqdm
from guidance import gen as guidance_gen
from guidance.models import Transformers
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM


tqdm.pandas()


########## global variables ########################
# hugging face model
model_id = "meta-llama/Llama-3.2-3B-Instruct"

HF_TOKEN = os.environ.get("HF_TOKEN") or True


tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True,token=HF_TOKEN)
model = model.to('cuda')
model = model.eval()



def construct_modified_query_base_llm(query):
    prompt_template = "Only use results from the genomic data commons in your response and provide frequencies as a percentage. Only report the final response."
    modified_query = query + prompt_template
    return modified_query


@spaces.GPU(duration=20)
def generate_percentage_response(modified_query, model, tok):
    set_seed(1042)
    regex = "The final response is: \d*\.\d*%"
    lm = Transformers(model=model, tokenizer=tok)
    lm += modified_query
    lm += guidance_gen(
        "pct_response",
        n=1,
        temperature=0,
        max_tokens=40,
        regex=regex,
    )
    return lm["pct_response"]




def batch_test(query):
    modified_query = construct_modified_query_base_llm(query)
    try:
        llama_base_output = generate_percentage_response(modified_query, model, tok)
    except Exception as e:
        print("unable to get qwen base output for q {} {}".format(query, e))
        llama_base_output = "no output obtained"

    return pd.Series(llama_base_output)


def main():

    # change to csv with questions
    questions = pd.read_csv("csvs/questions.csv")
    # questions = questions.head(n=3)
    print("getting llama-3B model responses on questions")
    questions["llama_base_output"] = questions["questions"].progress_apply(
        lambda x: batch_test(x)
    )
    print("done generation questions , dumping to CSV")
    questions.to_csv("csvs/llama.tests.results.guidance.csv")


if __name__ == "__main__":
    main()

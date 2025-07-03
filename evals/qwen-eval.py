#!/usr/bin/env python3


import json
import os

import pandas as pd
from huggingface_hub import HfFolder
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration

# vllm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

tqdm.pandas()

guided_decoding_params = GuidedDecodingParams(regex="The final answer is: \d*\.\d*%")

sampling_params_with_constrained_decoding = SamplingParams(
    n=1,
    temperature=0,
    seed=1042,
    max_tokens=1000,
    # to try remove repetition
    repetition_penalty=1.2,
    guided_decoding=guided_decoding_params,
)

########## global variables ########################
# hugging face model
# https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct" # doesnt work, this is vision and language
# https://huggingface.co/Qwen/Qwen1.5-4B-Chat
model_id = "Qwen/Qwen1.5-4B-Chat"

llm = LLM(model=model_id, trust_remote_code=True, enforce_eager=True)

# hugging face token
with open("/opt/gpudata/aartiv/rag_rig/huggingface_token.txt", "r") as hf_token_file:
    HF_TOKEN = hf_token_file.read().strip()
HfFolder.save_token(HF_TOKEN)


def construct_modified_query_base_llm(query):
    prompt_template = "Only use results from the genomic data commons in your response and provide frequencies as a percentage. Only report the final response."
    modified_query = query + prompt_template
    return modified_query


def batch_test(query):
    modified_query = construct_modified_query_base_llm(query)
    try:
        qwen_base_output = (
            llm.generate(modified_query, sampling_params_with_constrained_decoding)[0]
            .outputs[0]
            .text
        )
    except Exception as e:
        print("unable to get qwen base output for q {}".format(query))
        qwen_base_output = "no output obtained"

    return pd.Series(qwen_base_output)


def main():
    questions = pd.read_csv("csvs/questions.csv")
    # questions = questions.head(n=2)
    print("getting qwen model responses on questions")
    questions["qwen_base_output"] = questions["questions"].progress_apply(
        lambda x: batch_test(x)
    )
    print("done generation questions , dumping to CSV")
    questions.to_csv("csvs/qwen_results.csv")


if __name__ == "__main__":
    main()

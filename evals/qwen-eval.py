#!/usr/bin/env python3

import os

import pandas as pd
from tqdm import tqdm

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
# https://huggingface.co/Qwen/Qwen1.5-4B-Chat
model_id = "Qwen/Qwen1.5-4B-Chat"

HF_TOKEN = os.environ.get("HF_TOKEN") or True

llm = LLM(model=model_id, trust_remote_code=True, enforce_eager=True, dtype='half', max_model_len=8192)


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
        print("unable to get qwen base output for q {} {}".format(query, e))
        qwen_base_output = "no output obtained"

    return pd.Series(qwen_base_output)


def main():

    # change to csv with questions
    questions = pd.read_csv("csvs/cnv_or_ssm_new_questions.csv")
    print("getting qwen model responses on questions")
    questions["qwen_base_output"] = questions["questions"].progress_apply(
        lambda x: batch_test(x)
    )
    print("done generation questions , dumping to CSV")
    questions.to_csv("csvs/cnv_or_ssm_qwen_results.csv")


if __name__ == "__main__":
    main()

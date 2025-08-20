#!/usr/bin/env python3
# experiment with different query augmented prompts for qag

import argparse
import os
import ast
from types import SimpleNamespace
import json
import re
import pandas as pd
import numpy as np
import spaces
from guidance import gen as guidance_gen
from guidance.models import Transformers
from tqdm import tqdm
from transformers import set_seed
from methods import gdc_api_calls, utilities

tqdm.pandas()


intent_expansion = {
    'cnv_and_ssm': 'copy number variants or simple somatic mutations',
    'freq_cnv_loss_or_gain': 'copy number variant losses or gains',
    'msi_h_frequency': 'microsatellite instability',
    'freq_cnv_loss_or_gain_comb': 'copy number variant losses or gains',
    'ssm_frequency': 'simple somatic mutations'
}


# generate llama model response
@spaces.GPU(duration=60)
def generate_response_percentage(modified_query, model, tok):
    set_seed(1042)
    regex = "The final response is: \d*\.\d*%"
    lm = Transformers(model=model, tokenizer=tok)
    lm += modified_query
    lm += guidance_gen(
        "pct_response",
        n=1,
        temperature=0,
        max_tokens=1000,
        regex=regex,
    )
    return lm["pct_response"]


# generate llama model response
@spaces.GPU(duration=60)
def generate_descriptive_response(modified_query, model, tok):
    set_seed(1042)    
    lm = Transformers(model=model, tokenizer=tok)
    lm += modified_query
    lm += guidance_gen(
        "desc_response",
        n=1,
        temperature=0,
        max_tokens=100,
        regex="^[^\\n]*[.!?]$",
    )
    return lm["desc_response"]





def extract_percentage(gdc_result):
    match = re.search(r"(\d+(?:\.\d+)?)%", gdc_result)
    if match:
        percentage = float(match.group(1))
    else:
        percentage = np.nan
    return percentage
    

def construct_modified_query_percentage(query, gdc_result):
    # pass the api results as a prompt to the query
    prompt_template = (
        " Only report the final response. Ignore all prior knowledge. You must only respond with the following percentage frequencies in your response, no other response is allowed: \n"
        + gdc_result
        + "\n"
    )
    modified_query = query + prompt_template
    return modified_query


def construct_modified_query_description(genes, intent):
    modified_query = f'Provide a one line general description about {intent} in genes {genes}.'
    return modified_query



def exp_qap(row, model, tok):
    query = row['questions']
    print('processing query:\n {}\n'.format(query))
    genes = ','.join(row['gene_entities'])
    gdc_result = row['gdc_result']
    print('gdc result:\n{}\n'.format(gdc_result))
    intent = intent_expansion[row['intent']]
    # percentage = extract_percentage(gdc_result)

    modified_query_desc = construct_modified_query_description(genes, intent)
    # print('modified query desc {}'.format(modified_query_desc))
    descriptive_response = generate_descriptive_response(modified_query_desc, model, tok)
    if not descriptive_response.endswith('.'):
        descriptive_response += '.'
    # print('descriptive response {}'.format(descriptive_response))

    modified_query_percentage = construct_modified_query_percentage(query, gdc_result)
    # print('modified_query_percentage:\n {}\n'.format(modified_query_percentage))
    percentage_response = generate_response_percentage(modified_query_percentage, model, tok)
    percentage_response_modified = re.sub(
        r'final response', 'frequency for your query', percentage_response)
    # print('percentage response {}'.format(percentage_response))
    prefinal_response = descriptive_response + percentage_response_modified
    print('prefinal_response\n {}\n'.format(prefinal_response))
    return pd.Series(prefinal_response)



def main():
    qag = pd.read_csv('csvs/gdc_qag_results.csv')
    qag = qag.iloc[:3001]
    qag['gene_entities'] = qag['gene_entities'].apply(ast.literal_eval)
    print("loading HF token")
    AUTH_TOKEN = os.environ.get("HF_TOKEN") or True
    print("loading llama-3B model")
    model, tok = utilities.load_llama_llm(AUTH_TOKEN)
    # qag = qag.head(n=3)
    print('construct prompt + prefinal response')
    qag['prefinal_response'] = qag.progress_apply(
        lambda row: exp_qap(row, model, tok), axis=1 
    )
    qag.to_csv('csvs/qag_tests_pt1.csv', index=0)


if __name__ == '__main__':
    main()

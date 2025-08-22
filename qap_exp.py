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
    'ssm_frequency': 'simple somatic mutations',
    'top_cases_counts_by_gene': 'copy number variants or simple somatic mutations'
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
        max_tokens=40,
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
        # regex="^[^\\n]*[.!?]$",
        regex="^[^\\n]*[.\S+]$",


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
    modified_query = f'Provide a one line general description about {intent} in genes {genes} in cancer.'
    return modified_query



def exp_qap(row, model, tok):
    query = row['questions']
    print('processing query:\n {}\n'.format(query))
    genes = ','.join(row['gene_entities'])
    gdc_result = row['gdc_result']
    print('gdc result:\n{}\n'.format(gdc_result))
    intent = intent_expansion[row['intent']]
    descriptive_prompt = construct_modified_query_description(genes, intent)
    descriptive_response = generate_descriptive_response(descriptive_prompt, model, tok)
    if not descriptive_response.endswith('.'):
        descriptive_response += '.'
    
    percentage_prompt = construct_modified_query_percentage(query, gdc_result)
    # print('modified_query_percentage:\n {}\n'.format(modified_query_percentage))
    percentage_response = generate_response_percentage(percentage_prompt, model, tok)
    percentage_response = re.sub(
        r'final response', 'frequency for your query', percentage_response)
    return pd.Series([descriptive_prompt, percentage_prompt, descriptive_response, percentage_response])



def postprocess_llm_description(tok, descriptive_response):
    try:
        num_tokens = len(tok.encode(descriptive_response))
        if num_tokens < 100:
            postprocessed_desc_response = descriptive_response
        else:
            response_list = re.split(r'\.(?!\d+%)', descriptive_response)
            postprocessed_desc_response = '.'.join(response_list[:-1])
    except Exception as e:
        print('unable to postprocess LLM gene description {}'.format(
            str(e)
        ))
        postprocessed_desc_response = 'unable to postprocess LLM gene description'

    return postprocessed_desc_response


def postprocess_percentage_response(
        gdc_qag_base_stat, gdc_result_percentage, gdc_qag_percentage_response):
    
    try:
        # check/confirm if gdc_qag_base_stat percentage == gdc_result_percentage
        # change it, if not
        if gdc_qag_base_stat != gdc_result_percentage:
            gdc_qag_base_stat = gdc_result_percentage
            final_gdc_qag_percentage_response = 'The frequency for your query is: {}%'.format(
                gdc_qag_base_stat)
        else:
            final_gdc_qag_percentage_response = gdc_qag_percentage_response
    except Exception as e:
        print('unable to postprocess percentage frequency {}'.format(
            str(e)
        ))
        final_gdc_qag_percentage_response = 'unable to postprocess percentage frequency'
    return final_gdc_qag_percentage_response




def postprocess_response(tok, row):
    # four goals:
    # goal 1:
    # check/confirm the results in gdc-qag percentage response
    # return a percentage response for gdc-qag
    # goal 2:
    # calculate deltas for llama and gdc_qag models
    # goal 3:
    # postprocess descriptive response
    # goal 4:
    # return concatenated final response from gdc_qag
    # (descriptive response + percentage response)

    pattern = r".*?(\d*\.\d*)%.*?"

    ###### various inputs ###############################

    try:
        # this is the result obtained in GDC-QAG via API
        gdc_result = row["gdc_result"]
    except Exception as e:
        print('GDC Result not found in gdc_qag output, returning nan {}'.format(
            str(e)
        ))
        gdc_result = np.nan

    try:
        # this is externally generated truth
        truth = row['truth']
    except Exception as e:
        print('truth frequency not found, returning nan {}'.format(
            str(e)
        ))
        truth = np.nan

    try:
        # this is the LLM generated response with freq, after seeing gdc_result
        gdc_qag_percentage_response = row['percentage_response']
    except Exception as e:
        print('LLM generated gdc_qag percentage response not found, returning nan {}'.format(
            str(e)
        ))
        gdc_qag_percentage_response = np.nan

    # llama-3B base output
    llama_base_output = row["llama_base_output"]

    try:
        # extract llama percentage from llama base output
        llama_base_stat = float(re.search(pattern, llama_base_output).group(1))
    except Exception as e:
        print('unable to extract llama base stat {}'.format(str(e)))
        llama_base_stat = np.nan
    
    try:
        # extract gdc_result percentage from gdc_result
        gdc_result_percentage = float(re.search(pattern, gdc_result).group(1))
    except Exception as e:
        print('unable to extract percentage from gdc result {}'.format(
            str(e)))
        gdc_result_percentage = np.nan
    
    try:
        # extract gdc_qag percentage from LLM response
        gdc_qag_base_stat = float(re.search(pattern, gdc_qag_percentage_response).group(1))
    except Exception as e:
        print('unable to extract percentage from gdc_qag percentage response {}'.format(
            str(e)))
        gdc_qag_base_stat = np.nan

    
    ############ postprocess LLM description + percentage ###############

    final_gdc_qag_desc_response = postprocess_llm_description(
        tok, row['descriptive_response']
    )

    final_gdc_qag_percentage_response = postprocess_percentage_response(
        gdc_qag_base_stat, gdc_result_percentage, gdc_qag_percentage_response
    )

    final_gdc_qag_response = final_gdc_qag_desc_response + final_gdc_qag_percentage_response

    return pd.Series(
        [
            llama_base_stat,
            gdc_qag_base_stat,
            final_gdc_qag_desc_response,
            final_gdc_qag_percentage_response,
            final_gdc_qag_response
        ]
    )




def main():
    qag = pd.read_csv('csvs/test_qap_postprocess.csv')
    qag['gene_entities'] = qag['gene_entities'].apply(ast.literal_eval)

    
    print("loading HF token")
    AUTH_TOKEN = os.environ.get("HF_TOKEN") or True
    print("loading llama-3B model")
    model, tok = utilities.load_llama_llm(AUTH_TOKEN)
    # qag = qag.iloc[:3001]
    qag = qag.head(n=3)
    qag[['descriptive_prompt', 'percentage_prompt', 
         'descriptive_response', 'percentage_response']] = qag.progress_apply(
        lambda row: exp_qap(row, model, tok), axis=1 
    )
    # postprocess descriptive response + percentage response
    qag[['llama_base_stat', 'gdc_qag_base_stat', 
         'final_gdc_qag_desc_response', 
         'final_gdc_qag_percentage_response', 
         'final_gdc_qag_response']] = qag.apply(lambda row: postprocess_response(tok, row), axis=1)

    print('completed, writing results')
    qag.to_csv('csvs/qag_tests_pt1_revised.csv', index=0)


if __name__ == '__main__':
    main()

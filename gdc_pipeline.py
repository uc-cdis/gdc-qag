#!/usr/bin/env python3
# QAG pipeline entry point script


import argparse
import os
import re
from types import SimpleNamespace
import json
import pandas as pd
import spaces
from guidance import gen as guidance_gen
from guidance.models import Transformers
from tqdm import tqdm
from transformers import set_seed
from methods import gdc_api_calls, utilities

tqdm.pandas()


def execute_api_call(
    intent,
    gene_entities,
    mutation_entities,
    cancer_entities,
    query,
    gdc_genes_mutations,
    project_mappings,
):
    if intent == "ssm_frequency":
        result, cancer_entities = utilities.get_ssm_frequency(
            gene_entities, mutation_entities, cancer_entities, project_mappings
        )
    elif intent == "top_mutated_genes_by_project":
        result = gdc_api_calls.get_top_mutated_genes_by_project(
            cancer_entities, top_k=10
        )
    elif intent == "most_frequently_mutated_gene":
        result = gdc_api_calls.get_top_mutated_genes_by_project(
            cancer_entities, top_k=1
        )
    elif intent == "freq_cnv_loss_or_gain":
        result, cancer_entities = gdc_api_calls.get_freq_cnv_loss_or_gain(
            gene_entities, cancer_entities, query, cnv_and_ssm_flag=False
        )
    elif intent == "msi_h_frequency":
        result, cancer_entities = gdc_api_calls.get_msi_frequency(cancer_entities)
    elif intent == "cnv_and_ssm":
        result, cancer_entities = utilities.get_freq_of_cnv_and_ssms(
            query, cancer_entities, gene_entities, gdc_genes_mutations
        )
    elif intent == "top_cases_counts_by_gene":
        result, cancer_entities = gdc_api_calls.get_top_cases_counts_by_gene(
            gene_entities, cancer_entities
        )
    elif intent == "project_summary":
        result = gdc_api_calls.get_project_summary(cancer_entities)
    else:
        result = "user intent not recognized, or use case not covered"
    return result, cancer_entities


# function to combine entities, intent and API call
def construct_and_execute_api_call(
    query, gdc_genes_mutations, project_mappings, intent_model, intent_tok
):
    print(
        "\nStep 1: Starting GDC-QAG on input natural language query:\n{}\n".format(
            query
        )
    )
    # Infer entities
    initial_cancer_entities = utilities.return_initial_cancer_entities(
        query, model="en_ner_bc5cdr_md"
    )

    if not initial_cancer_entities:
        try:
            initial_cancer_entities = utilities.return_initial_cancer_entities(
                query, model="en_core_sci_md"
            )
        except Exception as e:
            print("unable to guess cancer entities {}".format(str(e)))
            initial_cancer_entities = []

    cancer_entities = utilities.postprocess_cancer_entities(
        project_mappings, initial_cancer_entities=initial_cancer_entities, query=query
    )

    # if cancer entities is empty from above methods
    # return all projects
    if not cancer_entities:
        cancer_entities = list(project_mappings.keys())
    gene_entities = utilities.infer_gene_entities_from_query(query, gdc_genes_mutations)
    mutation_entities = utilities.infer_mutation_entities(
        gene_entities=gene_entities,
        query=query,
        gdc_genes_mutations=gdc_genes_mutations,
    )
    print("\nStep 2: Entity Extraction\n")
    print("gene entities {}".format(gene_entities))
    print("mutation entities {}".format(mutation_entities))
    print("cancer entities {}".format(cancer_entities))

    # infer user intent
    intent = utilities.infer_user_intent(query, intent_model, intent_tok)
    print("\nStep 3: Intent Inference:\n{}\n".format(intent))
    try:
        print("\nStep 4: API call builder for intent {}\n".format(intent))
        api_call_result, cancer_entities = execute_api_call(
            intent,
            gene_entities,
            mutation_entities,
            cancer_entities,
            query,
            gdc_genes_mutations,
            project_mappings,
        )
        # print('cancer_entities {}'.format(cancer_entities))
    except Exception as e:
        print("unable to process query {} {}".format(query, str(e)))
        api_call_result = []
        cancer_entities = []
    return SimpleNamespace(
        gdc_result=api_call_result,
        cancer_entities=cancer_entities,
        intent=intent,
        gene_entities=gene_entities,
        mutation_entities=mutation_entities,
    )


# generate llama model pct response
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


# generate llama model descriptive response
@spaces.GPU(duration=20)
def generate_descriptive_response(modified_query, model, tok):
    set_seed(1042)    
    lm = Transformers(model=model, tokenizer=tok)
    lm += modified_query
    lm += guidance_gen(
        "desc_response",
        n=1,
        temperature=0,
        max_tokens=100,
        regex="^[^\\n]*[.\S+]$",
    )
    return lm["desc_response"]



def batch_test(
    query,
    model,
    tok,
    gdc_genes_mutations,
    project_mappings,
    intent_model,
    intent_tok
):
    modified_query = utilities.construct_modified_query_base_llm(query)
    print(f"obtain baseline llama-3B response on modified query: {modified_query}")
    llama_base_output = generate_percentage_response(modified_query, model, tok)
    print(f"llama-3B baseline response: {llama_base_output}")
    try:
        result = construct_and_execute_api_call(
            query, gdc_genes_mutations, project_mappings, intent_model, intent_tok
        )
    except Exception as e:
        # unable to compute at this time, recheck
        result.gdc_result = []
        result.cancer_entities = []
    # if there is not a helper output for each unique cancer entity
    # log error to inspect and reprocess query later
    try:
        len(result.gdc_result) == len(result.cancer_entities)
    except Exception as e:
        msg = "there is not a unique gdc result for each unique \
    cancer entity in {}".format(
            query
        )
        print("exception {}".format(msg))
        result.gdc_result = []
        result.cancer_entities = []

    return pd.Series(
        [
            llama_base_output,
            result.gdc_result,
            result.cancer_entities,
            result.intent,
            result.gene_entities,
            result.mutation_entities,
        ]
    )


def setup_args():
    parser = argparse.ArgumentParser()
    # add functionality to either pass in a file with questions or a single question
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-file",
        dest="input_file",
        help="path to input file with questions. input file should contain one column named questions, with each question on one line",
    )
    group.add_argument("--question", dest="question", help="a single question string")
    return parser.parse_args()


def get_prefinal_response(row, model, tok):
    try:
        query = row["questions"]
        genes = ','.join(row['gene_entities'])
        gdc_result = row["gdc_result"]
    except Exception as e:
        print(f"unable to retrieve query: {query} or gdc_result: {gdc_result}")
    
    intent = utilities.intent_expansion[row['intent']]

    print("\nStep 6: Construct LLM prompts for llama-3B\n")
    descriptive_prompt = utilities.construct_modified_query_description(genes, intent)
    percentage_prompt = utilities.construct_modified_query_percentage(query, gdc_result)
    
    print("\nStep 7: Generate LLM response R on query augmented prompts\n")
    descriptive_response = generate_descriptive_response(descriptive_prompt, model, tok)
    if not descriptive_response.endswith('.'):
        descriptive_response += '.'
    
    percentage_response = generate_percentage_response(percentage_prompt, model, tok)
    percentage_response = re.sub(
        r'final response', 'frequency for your query', percentage_response)
    return pd.Series([
        descriptive_prompt, percentage_prompt, 
        descriptive_response, percentage_response
        ])
    


def setup_models_and_data():
    # from env
    print("loading HF token")
    AUTH_TOKEN = os.environ.get("HF_TOKEN") or True

    print("getting gdc project information")
    # retrieve and load GDC project mappings
    project_mappings = gdc_api_calls.get_gdc_project_ids(start=0, stop=86)

    print("loading gdc genes and mutations")
    gdc_genes_mutations = utilities.load_gdc_genes_mutations_hf(AUTH_TOKEN)

    print("loading llama-3B model")
    model, tok = utilities.load_llama_llm(AUTH_TOKEN)

    print('loading intent model')
    intent_model, intent_tok = utilities.load_intent_model_hf(AUTH_TOKEN)
    return SimpleNamespace(
        project_mappings=project_mappings,
        gdc_genes_mutations=gdc_genes_mutations,
        model=model,
        tok=tok,
        intent_model=intent_model,
        intent_tok=intent_tok
    )



@utilities.timeit
def execute_pipeline(
    df, gdc_genes_mutations, model, 
    tok, intent_model, intent_tok, 
    project_mappings, output_file_prefix
):
    
    df[
        [
            "llama_base_output",
            "gdc_result",
            "cancer_entities",
            "intent",
            "gene_entities",
            "mutation_entities",
        ]
    ] = df["questions"].progress_apply(
        lambda x: batch_test(
            x,
            model,
            tok,
            gdc_genes_mutations,
            project_mappings,
            intent_model,
            intent_tok
        )
    )

    # retain responses with gdc_result

    df_exploded = df.explode('gdc_result', ignore_index=True)

    df_exploded[["descriptive_prompt", "percentage_prompt", "descriptive_response", "percentage_response"]] = (
        df_exploded.progress_apply(
            lambda x: get_prefinal_response(x, model, tok), axis=1
        )
    )

    ### final check and confirmation 
    print("\nStep 8: Final check and confirmation\n")
    
    # postprocess descriptive response + percentage response
    df_exploded[
        [
            'llama_base_stat', 
            'gdc_qag_base_stat', 
            'final_gdc_qag_desc_response', 
            'final_gdc_qag_percentage_response', 
            'final_gdc_qag_response'
        ]
    ] = df_exploded.progress_apply(
        lambda row: utilities.postprocess_response(tok, row), axis=1)
    
    final_columns = utilities.get_final_columns()
    result = df_exploded[final_columns].copy()

    result.rename(
        columns={
            "llama_base_output": "llama-3B baseline output",
            "descriptive_prompt": "Descriptive prompt",
            "percentage_prompt": "Query augmented prompt",
            "gdc_result": "GDC Result",
            "gdc_qag_base_stat": "GDC-QAG frequency",
            "llama_base_stat": "llama-3B baseline frequency",
            "final_gdc_qag_response": "Query augmented generation",
            "intent": "Intent",
            "cancer_entities": "Cancer entities",
            "gene_entities": "Gene entities",
            "mutation_entities": "Mutation entities",
            "questions": "Question"
        },
        inplace=True,
    )
    result.index = ["GDC-QAG results"] * len(result)
    print(
        "Query Augmented Generation final response {}".format(
            "\n".join(result["Query augmented generation"].astype(str))
        )
    )
    print("completed")
    
    if output_file_prefix:
        final_output = os.path.join("csvs", output_file_prefix + ".results.csv")
        print("writing final results to {}".format(final_output))
        result.to_csv(final_output, index=0)
    else:
        print(json.dumps(result.T.to_dict(), indent=2))
    print('completed')
    return result



def main():
    args = setup_args()
    input_file = args.input_file or None
    question = args.question or None

    qag_requirements = setup_models_and_data()

    if input_file:
        df = pd.read_csv(input_file)
        output_file_prefix = os.path.basename(input_file).split(".")[0]
        execute_pipeline(
            df, 
            qag_requirements.gdc_genes_mutations,
            qag_requirements.model,
            qag_requirements.tok,
            qag_requirements.intent_model,
            qag_requirements.intent_tok,
            qag_requirements.project_mappings,
            output_file_prefix
        )
    elif question:
        df = pd.DataFrame({"questions": [question]})
        execute_pipeline(
            df, 
            qag_requirements.gdc_genes_mutations,
            qag_requirements.model,
            qag_requirements.tok,
            qag_requirements.intent_model,
            qag_requirements.intent_tok,
            qag_requirements.project_mappings,
            output_file_prefix=None
        )


if __name__ == "__main__":
    main()

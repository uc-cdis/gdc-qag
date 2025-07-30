#!/usr/bin/env python3
# QAG pipeline entry point script


import argparse
import os
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
    print("query:\n{}\n".format(query))
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

    print("gene entities {}".format(gene_entities))
    print("mutation entities {}".format(mutation_entities))
    print("cancer entities {}".format(cancer_entities))

    # infer user intent
    intent = utilities.infer_user_intent(query, intent_model, intent_tok)
    print("user intent:\n{}\n".format(intent))
    try:
        api_call_result, cancer_entities = execute_api_call(
            intent,
            gene_entities,
            mutation_entities,
            cancer_entities,
            query,
            gdc_genes_mutations,
            project_mappings,
        )
        print("api_call_result {}".format(api_call_result))
        # print('cancer_entities {}'.format(cancer_entities))
    except Exception as e:
        print("unable to process query {} {}".format(query, str(e)))
        api_call_result = []
        cancer_entities = []
    return SimpleNamespace(
        helper_output=api_call_result,
        cancer_entities=cancer_entities,
        intent=intent,
        gene_entities=gene_entities,
        mutation_entities=mutation_entities,
    )


# generate llama model response
@spaces.GPU(duration=60)
def generate_response(modified_query, model, tok):
    set_seed(1042)
    regex = "The final answer is: \d*\.\d*%"
    lm = Transformers(model=model, tokenizer=tok)
    lm += modified_query
    lm += guidance_gen(
        "gen_response",
        n=1,
        temperature=0,
        max_tokens=1000,
        # to try remove repetition, this is not a param in guidance
        # repetition_penalty=1.2,
        regex=regex,
    )
    return lm["gen_response"]


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
    llama_base_output = generate_response(modified_query, model, tok)
    try:
        result = construct_and_execute_api_call(
            query, gdc_genes_mutations, project_mappings, intent_model, intent_tok
        )
    except Exception as e:
        # unable to compute at this time, recheck
        result.helper_output = []
        result.cancer_entities = []
    # if there is not a helper output for each unique cancer entity
    # log error to inspect and reprocess query later
    try:
        len(result.helper_output) == len(result.cancer_entities)
    except Exception as e:
        msg = "there is not a unique helper output for each unique \
    cancer entity in {}".format(
            query
        )
        print("exception {}".format(msg))
        result.helper_output = []
        result.cancer_entities = []

    return pd.Series(
        [
            llama_base_output,
            result.helper_output,
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
        helper_output = row["helper_output"]
    except Exception as e:
        print(f"unable to retrieve query: {query} or helper_output: {helper_output}")
    modified_query = utilities.construct_modified_query(query, helper_output)
    prefinal_llama_with_helper_output = generate_response(modified_query, model, tok)
    return pd.Series([modified_query, prefinal_llama_with_helper_output])


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
    print("starting pipeline")

    # queries input file
    print(f"running test on input {df}")
    df[
        [
            "llama_base_output",
            "helper_output",
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

    # retain responses with helper output

    df_exploded = df.explode('helper_output', ignore_index=True)
    df_exploded[["modified_prompt", "pre_final_llama_with_helper_output"]] = (
        df_exploded.progress_apply(
            lambda x: get_prefinal_response(x, model, tok), axis=1
        )
    )

    ### postprocess response
    print("postprocessing response")
    df_exploded[
        [
            "llama_base_stat",
            "delta_llama",
            "value_changed",
            "ground_truth_stat",
            "generated_stat_prefinal",
            "delta_prefinal",
            "generated_stat_final",
            "delta_final",
            "final_response",
        ]
    ] = df_exploded.progress_apply(
        lambda x: utilities.postprocess_response(x), axis=1
    )

    final_columns = utilities.get_final_columns()

    if output_file_prefix:
        final_output = os.path.join("csvs", output_file_prefix + ".results.csv")
        print("writing final results to {}".format(final_output))
        df_exploded.to_csv(final_output, columns=final_columns)
        result = df_exploded[final_columns]
    else:
        result = df_exploded[final_columns]
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

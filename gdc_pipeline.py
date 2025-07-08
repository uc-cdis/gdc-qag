#!/usr/bin/env python3
# GDC-RAG pipeline entry point script

# import libraries
import argparse
import inspect
import json
import os
from types import SimpleNamespace

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from methods import gdc_api_calls, utilities
from methods.tools import (
    construct_tools_list,
    construct_tools_params_dict,
    get_cnv_and_ssm,
    get_freq_cnv_loss_or_gain,
    get_msi_h_frequency,
    get_ssm_frequency,
    get_top_cases_counts_by_gene,
    tool_functions_list
)

tqdm.pandas()

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


def execute_api_call(
    intent,
    gene_entities,
    mutation_entities,
    cancer_entities,
    query,
    gdc_genes_mutations,
    project_mappings,
):
    # TO-DO
    # return a intent not recognized response if no intent match
    # content = "Get results for intent {}, populate all default values in tool arguments. When generating tool arguments, ensure that any list-type arguments are returned as python lists, not strings".format(intent)
    content = 'Get results for intent {}'.format(intent)
    tools = construct_tools_list()
    print('tools {}'.format(tools))
    print('completed constructing tools list')
    tools_params_dict = construct_tools_params_dict(
        intent, gene_entities, mutation_entities, cancer_entities,
        project_mappings, query, gdc_genes_mutations)
    
    print('sending request to client for content {}'.format(content))
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": content}],
        tools=tools,
        tool_choice="auto",
        temperature=0,
        seed=42,
        n=1
    )
    print('received response {}'.format(response))
    tool_call = response.choices[0].message.tool_calls[0].function
    print(f"Function called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    
    # inspect signature and manually inject args
    sig = inspect.signature(tool_functions_list[tool_call.name])
    arg_names = list(sig.parameters.keys())
    print('arg_names {}'.format(arg_names))
    args_dict = {name: tools_params_dict[name] for name in arg_names}
    # override tool call args with args_dict
    tool_call.arguments = json.dumps(args_dict)
    # execute
    result, cancer_entities = tool_functions_list[tool_call.name](
        **json.loads(tool_call.arguments)
    )
    return result, cancer_entities


# function to combine entities, intent and API call
def construct_api_call(query, intent_model_path, gdc_genes_mutations, project_mappings):
    print("query:\n{}\n".format(query))
    # Infer entities
    initial_cancer_entities = utilities.return_initial_cancer_entities(
        query, model="en_ner_bc5cdr_md"
    )
    # print('initial cancer entities {}'.format(initial_cancer_entities))
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
    intent = utilities.infer_user_intent(query, intent_model_path)
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


def batch_test(
    query,
    llm,
    sampling_params,
    intent_model_path,
    gdc_genes_mutations,
    project_mappings,
):
    modified_query = utilities.construct_modified_query_base_llm(query)
    llama_base_output = llm.generate(modified_query, sampling_params)[0].outputs[0].text
    try:
        result = construct_api_call(
            query, intent_model_path, gdc_genes_mutations, project_mappings
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
    parser.add_argument(
        "--input-file",
        dest="input_file",
        help="path to input file with questions. input file should contain one column named questions, with each question on one line",
        required=True,
    )
    parser.add_argument(
        "--intent-model-path",
        dest="intent_model_path",
        help="path to trained BERT model for user intent",
        default="/opt/gpudata/aartiv/qag/query_intent_model.pt",
    )
    parser.add_argument(
        "--path-to-gdc-genes-mutations-file",
        dest="path_to_gdc_genes_mutations_file",
        help="path to dumped genes and mutations info from gdc",
        default="/opt/gpudata/aartiv/qag/gdc_genes_mutations.json",
    )
    parser.add_argument(
        "--hf-token-path",
        dest="hf_token_path",
        help="path to hugging face token",
        default="/opt/gpudata/aartiv/qag/huggingface_token.txt",
    )
    return parser.parse_args()


def execute_pipeline(
    input_file, intent_model_path, path_to_gdc_genes_mutations, hf_token_path
):
    # load hf token
    print("starting pipeline")

    print("loading HF token")
    utilities.set_hf_token(hf_token_path)

    print("getting gdc project information")
    # retrieve and load GDC project mappings
    project_mappings = gdc_api_calls.get_gdc_project_ids(start=0, stop=86)
    # define models

    print("loading gdc genes and mutations")
    gdc_genes_mutations = utilities.load_gdc_genes_mutations(
        path_to_gdc_genes_mutations
    )

    print("loading guided decoding params")
    llm, sampling_params = utilities.load_llama_llm()

    # queries input file
    print("running batch test on input queries file {}".format(input_file))
    llm_eval_dataset = pd.read_csv(input_file)
    llm_eval_dataset[
        [
            "llama_base_output",
            "helper_output",
            "cancer_entities",
            "intent",
            "gene_entities",
            "mutation_entities",
        ]
    ] = llm_eval_dataset["questions"].progress_apply(
        lambda x: batch_test(
            x,
            llm,
            sampling_params,
            intent_model_path,
            gdc_genes_mutations,
            project_mappings,
        )
    )

    # get RAG response with helper output
    llm_eval_dataset["len_helper"] = llm_eval_dataset["helper_output"].apply(
        lambda x: len(x)
    )
    llm_eval_dataset_filtered = llm_eval_dataset[llm_eval_dataset["len_helper"] != 0]
    llm_eval_dataset_filtered["len_ce"] = llm_eval_dataset_filtered[
        "cancer_entities"
    ].apply(lambda x: len(x))
    # retain rows where one response is retrieved for each cancer entity
    llm_eval_dataset_filtered["ce_eq_helper"] = llm_eval_dataset_filtered.apply(
        lambda x: x["len_ce"] == x["len_helper"], axis=1
    )
    llm_eval_dataset_filtered = llm_eval_dataset_filtered[
        llm_eval_dataset_filtered["ce_eq_helper"]
    ]

    llm_eval_dataset_filtered_exploded = llm_eval_dataset_filtered.explode(
        ["helper_output", "cancer_entities"], ignore_index=True
    )
    input_file_prefix = os.path.basename(input_file).split(".")[0]
    exploded_per_cancer_output = os.path.join(
        "csvs", input_file_prefix + ".intermediate.csv"
    )

    print(
        "writing one response per cancer project to {}".format(
            exploded_per_cancer_output
        )
    )
    llm_eval_dataset_filtered_exploded.to_csv(exploded_per_cancer_output)

    llm_eval_dataset_filtered_exploded[
        ["modified_prompt", "pre_final_llama_with_helper_output"]
    ] = llm_eval_dataset_filtered_exploded.progress_apply(
        lambda x: utilities.get_prefinal_response(x, llm, sampling_params), axis=1
    )
    prefinal_output = os.path.join("csvs", input_file_prefix + ".prefinal.csv")
    print(
        "writing prefinal output (without postprocessing) to {}".format(prefinal_output)
    )

    ### postprocess response
    print("postprocessing response")
    llm_eval_dataset_filtered_exploded[
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
    ] = llm_eval_dataset_filtered_exploded.progress_apply(
        lambda x: utilities.postprocess_response(x), axis=1
    )
    final_output = os.path.join("csvs", input_file_prefix + ".results.csv")
    print("writing final results to {}".format(final_output))
    final_columns = utilities.get_final_columns()
    llm_eval_dataset_filtered_exploded.to_csv(final_output, columns=final_columns)

    print("completed")


def main():
    args = setup_args()
    input_file = args.input_file
    intent_model_path = args.intent_model_path
    path_to_gdc_genes_mutations = args.path_to_gdc_genes_mutations_file
    hf_token_path = args.hf_token_path
    execute_pipeline(
        input_file, intent_model_path, path_to_gdc_genes_mutations, hf_token_path
    )


if __name__ == "__main__":
    main()

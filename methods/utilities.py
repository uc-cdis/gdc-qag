#!/usr/bin/env python3
# various utility functions employed by the pipeline
import json
import re
from functools import reduce

import numpy as np
import pandas as pd
import spacy
import torch
#from huggingface_hub import HfFolder
from transformers import BertTokenizer

# vllm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from methods import gdc_api_calls


def load_llama_llm():
    # hugging face model
    # https://huggingface.co/blog/llama32
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    guided_decoding_params = GuidedDecodingParams(
        regex="The final answer is: \d*\.\d*%"
    )
    # add dtype=half if not using A100 (e.g. titan or V100)
    llm = LLM(
        model=model_id,
        dtype="half",
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=8184
    )
    sampling_params_with_constrained_decoding = SamplingParams(
        n=1,
        temperature=0,
        seed=1042,
        max_tokens=1000,
        # to try remove repetition
        repetition_penalty=1.2,
        guided_decoding=guided_decoding_params,
    )
    return llm, sampling_params_with_constrained_decoding


def set_guided_decoding_params():
    guided_decoding_params = GuidedDecodingParams(
        regex="The final answer is: \d*\.\d*%"
    )
    return guided_decoding_params


def load_gdc_genes_mutations(path_to_gdc_genes_mutations_file):
    gdc_genes_mutations = json.load(open(path_to_gdc_genes_mutations_file))
    return gdc_genes_mutations


def load_intent_model(intent_model_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # model = torch.load('/opt/gpudata/aartiv/rag_rig/query_intent_model.pt')
    model = torch.load(intent_model_path)
    return model, tokenizer


def infer_user_intent(query, intent_model_path):
    model, tokenizer = load_intent_model(intent_model_path)
    intent_labels = {
        "ssm_frequency": 0.0,
        "msi_h_frequency": 1.0,
        "freq_cnv_loss_or_gain": 2.0,
        "top_cases_counts_by_gene": 3.0,
        "cnv_and_ssm": 4.0,
    }
    # set device and load both model and query on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # pass tokenized input through the model
    outputs = model(**inputs)
    # print('output logits {}'.format(outputs))
    # outputs are logits, need to apply softmax to convert to probs
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    # print('probs: {}'.format(probs))
    predicted_label = torch.argmax(probs, dim=1).item()
    for k, v in intent_labels.items():
        if v == predicted_label:
            # print('predicted label: {}\n'.format(k))
            return k


def construct_modified_query_base_llm(query):
    prompt_template = "Only use results from the genomic data commons in your response and provide frequencies as a percentage. Only report the final response."
    modified_query = query + prompt_template
    return modified_query


def construct_modified_query(query, helper_output):
    # pass the api results as a prompt to the query
    prompt_template = (
        " Only report the final response. Ignore all prior knowledge. You must only respond with the following percentage frequencies in your response, no other response is allowed: \n"
        + helper_output
        + "\n"
    )
    modified_query = query + prompt_template
    return modified_query


def get_total_case_counts(ssm_counts_by_project):
    for project in ssm_counts_by_project.keys():
        total_case_count = gdc_api_calls.get_available_ssm_data_for_project(project)
        ssm_counts_by_project[project]["total_case_counts"] = total_case_count
    return ssm_counts_by_project


def calculate_ssm_frequency(ssm_statistics, cancer_entities, project_mappings):
    if not cancer_entities:
        cancer_entities = list(project_mappings.keys())
    pre_final_ssm_frequency = {}
    ssm_frequency = {}
    for project in ssm_statistics.keys():
        freq = (
            ssm_statistics[project]["ssm_counts"]
            / ssm_statistics[project]["total_case_counts"]
        )
        pre_final_ssm_frequency[project] = {"frequency": round(freq * 100, 2)}

    for c in cancer_entities:
        if c in pre_final_ssm_frequency:
            ssm_frequency[c] = pre_final_ssm_frequency[c]
        else:
            ssm_frequency[c] = {"frequency": 0.0}
    return ssm_frequency


def calculate_joint_ssm_frequency_v2(ssm_statistics, mutation_list, cancer_entities):
    # stores the result for all cancers
    joint_ssm_frequency = {}
    # initialize joint_freq by cancer entities
    joint_ssm_frequency_for_cancer = {}
    for c in cancer_entities:
        joint_ssm_frequency_for_cancer[c] = {}
        joint_ssm_frequency_for_cancer[c] = {"joint_frequency": 0.0}

    projects_with_mutation = [
        set(ssm_statistics[mutation].keys()) for mutation in mutation_list
    ]
    overlapping_projects_with_mutation = list(
        reduce(lambda x, y: x & y, projects_with_mutation)
    )
    for project in overlapping_projects_with_mutation:
        cases_with_mutation = [
            set(ssm_statistics[mutation][project]["case_id_list"])
            for mutation in mutation_list
        ]
        shared_cases = list(reduce(lambda x, y: x & y, cases_with_mutation))
        # print('shared cases, len shared cases {} {}'.format(shared_cases, len(shared_cases)))
        if shared_cases:
            if project not in joint_ssm_frequency:
                joint_ssm_frequency[project] = {}
            total_case_counts = gdc_api_calls.get_available_ssm_data_for_project(
                project
            )
            joint_frequency = len(shared_cases) / total_case_counts
            # print('shared_cases {}'.format(shared_cases))
            # print('joint freq {}'.format(joint_frequency))
            joint_ssm_frequency[project]["joint_frequency"] = round(
                joint_frequency * 100, 2
            )
    # filter for specific cancer type and return
    for c in cancer_entities:
        if c in joint_ssm_frequency:
            joint_ssm_frequency_for_cancer[c]["joint_frequency"] = joint_ssm_frequency[
                c
            ]["joint_frequency"]
    return joint_ssm_frequency_for_cancer


def flatten_ssm_results_to_text(result, result_type):
    result_text = []
    if result_type == "joint_frequency":
        for k, v in result.items():
            if k == "joint_frequency":
                for k2, v2 in v.items():
                    result_text.append(
                        "joint frequency in {} is {}%".format(k2, v2["joint_frequency"])
                    )
    else:
        for k, v in result.items():
            if k != "joint_frequency":
                for k2, v2 in v.items():
                    result_text.append(
                        "The frequency of {} in {} is {}%".format(
                            k, k2, v2["frequency"]
                        )
                    )
    return result_text


def get_ssm_frequency(
    gene_entities, mutation_entities, cancer_entities, project_mappings
):
    ssm_statistics = {}
    mutation_list = []
    result = {}
    # to match the genes with mutations
    if len(mutation_entities) > len(gene_entities):
        gene_entities = gene_entities * len(mutation_entities)
    # print('gene entities {}'.format(gene_entities))
    for gene, mutation in zip(gene_entities, mutation_entities):
        mutation_name = "_".join([gene, mutation])
        # print('computing frequency of {}'.format(mutation_name))
        mutation_list.append(mutation_name)
        ssm_id = gdc_api_calls.get_ssm_id(gene, mutation)
        ssm_counts_by_project = gdc_api_calls.get_ssm_counts(ssm_id)
        ssm_statistics[mutation_name] = get_total_case_counts(ssm_counts_by_project)
        # full_result for all cancer entities
        # test code for generalizability to multiple cancer entities
        # full_result format is {'project1': {'frequency': }, 'project2': {'frequency':}, 'projectn': {'frequency':}}
        full_result = calculate_ssm_frequency(
            ssm_statistics[mutation_name], cancer_entities, project_mappings
        )
        # result format:
        """
    { 
      'gene_mutation': # e.g. JAK2_V617F
      {
        'project1': {'frequency': }, 
        'project2': {'frequency':}, 
        'projectn': {'frequency':}
      }
    }
    'project1': {'frequency': }, 'project2': {'frequency':}
    """
        result[mutation_name] = {
            k: v for k, v in full_result.items() if k in cancer_entities
        }
        # if no entity match to specific gdc projects, return all
        if not result[mutation_name].values():
            result[mutation_name] = full_result
    # print('API result ssm freq {}'.format(result))
    # final cancer entities
    for k, v in result.items():
        cancer_entities = list(v.keys())
    # print('ssm freq cancer entities {}'.format(cancer_entities))
    # print('mutation list {}'.format(mutation_list))
    # only supporting for two mutations atm
    if len(mutation_list) > 1:
        # print('computing joint frequency')
        result["joint_frequency"] = calculate_joint_ssm_frequency_v2(
            ssm_statistics, mutation_list=mutation_list, cancer_entities=cancer_entities
        )
        result_text = flatten_ssm_results_to_text(result, result_type="joint_frequency")
    else:
        result["joint_frequency"] = 0
        result_text = flatten_ssm_results_to_text(
            result, result_type="single_frequency"
        )
    # print('result_text {}'.format(result_text))
    return result_text, cancer_entities


def decompose_mutation_and_cnv(query, match_term, gdc_genes_mutations):
    decompose_result = {}
    genes = [g for g in query.split(" ") if g in gdc_genes_mutations.keys()]
    # query must have cnv first, followed by mutation
    cnv_gene_name, mut_gene_name = genes[0], genes[1]
    # print('cnv_gene_name, mut_gene_name {} {}'.format(
    #  cnv_gene_name, mut_gene_name))
    decompose_result["cnv_and_ssm"] = True
    decompose_result["cnv_gene"] = cnv_gene_name
    decompose_result["mut_gene"] = mut_gene_name
    decompose_result["cnv_change_type"] = match_term
    return decompose_result


def get_freq_of_cnv_and_ssms(
    query, cancer_entities, gene_entities, gdc_genes_mutations
):
    lc_query = query.lower()
    match_term = ""
    cnv_terms = [
        "amplification",
        "deletion",
        "loss",
        "gain",
        "homozygous deletion",
        "heterozygous deletion",
    ]
    for term in cnv_terms:
        if term in lc_query:
            match_term = term
    # print('match_term {}'.format(match_term))
    if match_term:
        decompose_result = decompose_mutation_and_cnv(
            query, match_term, gdc_genes_mutations
        )
        # print('decompose result {}'.format(decompose_result))
        result, cancer_entities = gdc_api_calls.run_cnv_ssm_api(
            decompose_result, cancer_entities, query
        )
        # print('result {}'.format(result))
    else:
        # no specific match terms, return freq of cnvs + ssm
        result, cancer_entities = gdc_api_calls.get_top_cases_counts_by_gene(
            gene_entities, cancer_entities
        )
    return result, cancer_entities


def return_initial_cancer_entities(query, model):
    nlp = spacy.load(model)
    doc = nlp(query)
    result = doc.ents
    initial_cancer_entities = [e.text for e in result if e.label_ == "DISEASE"]
    return initial_cancer_entities


def infer_gene_entities_from_query(query, gdc_genes_mutations):
    entities = []
    # gene recognition with simple dict-based method
    for g in gdc_genes_mutations.keys():
        if (g in query) and (g in query.split(" ")):
            entities.append(g)
    return entities


def check_if_project_id_in_query(project_list, query):
    # check if mention of project keys
    # e.g. TCGA-BRCA in query
    final_entities = [
        potential_ce
        for potential_ce in query.split(" ")
        if potential_ce in project_list
    ]
    return final_entities


def proj_id_and_partial_match(query, project_mappings, initial_cancer_entities):
    final_entities = []
    if initial_cancer_entities:
        # print('checking for full match between initial cancer entities and GDC project descriptions')
        # check for match with project_mapping values
        #  e.g. match "ovarian serous cystadenocarcinoma" to TCGA-OV project
        for ic in initial_cancer_entities:
            for k, v in project_mappings.items():
                for c in v:
                    if ic in c.lower():
                        # print('found!!! {} {}'.format(ic, c.lower()))
                        final_entities.append(k)
    else:
        # print('no initial cancer entities, check for full match between query terms and GDC project descriptions')
        for term in query.lower().split(" "):
            for k, v in project_mappings.items():
                for c in v:
                    if term in c.lower():
                        # print('found!!! {} {}'.format(ic, c.lower()))
                        final_entities.append(k)
    return list(set(final_entities))


def postprocess_cancer_entities(project_mappings, initial_cancer_entities, query):
    # print('initial cancer entities {}'.format(initial_cancer_entities))
    project_list = project_mappings.keys()
    # print('check if GDC project-id mentioned in query')
    final_entities = check_if_project_id_in_query(project_list, query)
    if final_entities:
        return final_entities
    else:
        if initial_cancer_entities:
            # first query GDC projects endpt
            # print('test 1 (w/ initial entities): querying GDC projects endpt for project_id')
            gdc_project_match = gdc_api_calls.map_cancer_entities_to_project(
                initial_cancer_entities, project_mappings
            )
            # print('mapped projects to ids {}'.format(gdc_project_match))
            if gdc_project_match.values():
                final_entities = list(gdc_project_match.values())
            if not final_entities:
                # print('test 2 (w/ initial entities): no result from GDC projects endpt, check for matches '
                #    'between query terms and gdc project_mappings')
                final_entities = proj_id_and_partial_match(
                    query, project_mappings, initial_cancer_entities
                )
        else:
            # no initial_cancer_entities
            # check project_mappings keys/values for matches with query terms
            # print('test 3 (w/o initial entities): no result from GDC projects endpt, check for matches '
            #      'between query terms and gdc project_mappings')
            final_entities = proj_id_and_partial_match(
                query, project_mappings, initial_cancer_entities
            )
    return final_entities


def infer_mutation_entities(gene_entities, query, gdc_genes_mutations):
    mutation_entities = []
    for g in gene_entities:
        for m in gdc_genes_mutations[g]:
            if m in query:
                mutation_entities.append(m)
    return mutation_entities


def get_prefinal_response(row, llm, sampling_params):
    try:
        query = row["questions"]
        helper_output = row["helper_output"]
    except Exception as e:
        print(f"unable to retrieve query: {query} or helper_output: {helper_output}")
    modified_query = construct_modified_query(query, helper_output)
    prefinal_llama_with_helper_output = (
        llm.generate(modified_query, sampling_params)[0].outputs[0].text
    )
    return pd.Series([modified_query, prefinal_llama_with_helper_output])


def postprocess_response(row):
    value_changed = "no"
    pattern = r".*?(\d*\.\d*)%.*?"
    delta_final = np.nan
    delta_prefinal = np.nan
    generated_stat_final = np.nan

    try:
        helper_output = row["helper_output"]
    except Exception as e:
        # print('unable to generate helper output, returning nan')
        return pd.Series(["np.nan"] * 8)

    pre_final_response = row["pre_final_llama_with_helper_output"]
    llama_base_output = row["llama_base_output"]

    try:
        llama_base_stat = float(re.search(pattern, llama_base_output).group(1))
    except Exception as e:
        # print('unable to extract llama base stat {}'.format(str(e)))
        llama_base_stat = np.nan
    try:
        generated_stat_prefinal = float(re.search(pattern, pre_final_response).group(1))
    except Exception as e:
        # print('unable to extract generated stat {}'.format(str(e)))
        generated_stat_prefinal = np.nan

    try:
        ground_truth_stat = float(re.search(pattern, helper_output).group(1))
    except Exception as e:
        # print('unable to extract ground truth stat {}'.format(str(e)))
        ground_truth_stat = np.nan

    try:
        delta_llama = llama_base_stat - ground_truth_stat
    except Exception as e:
        # print('unable to calculate delta_llama {}'.format(str(e)))
        delta_llama = np.nan

    if not np.isnan(generated_stat_prefinal) and not np.isnan(ground_truth_stat):
        delta_prefinal = generated_stat_prefinal - ground_truth_stat
        if delta_prefinal != 0.0:
            final_response = "The final answer is: {}%".format(ground_truth_stat)
            value_changed = "yes"
        else:
            final_response = pre_final_response
        generated_stat_final = float(re.search(pattern, final_response).group(1))
        delta_final = generated_stat_final - ground_truth_stat
    else:
        final_response = "unable to postprocess, check generated or truth stat"
        value_changed = "na"
    """
  print('check if all values are populated:\n')
  print('delta_llama {}'.format(delta_llama))
  print('value_changed {}'.format(value_changed))
  print('ground_truth_stat {}'.format(ground_truth_stat))
  print('generated_stat_prefinal {}'.format(generated_stat_prefinal))
  print('delta_prefinal {}'.format(delta_prefinal))
  print('generated_stat_final {}'.format(generated_stat_final))
  print('delta_final {}'.format(delta_final))
  print('final_response {}'.format(final_response))
  """
    return pd.Series(
        [
            llama_base_stat,
            delta_llama,
            value_changed,
            ground_truth_stat,
            generated_stat_prefinal,
            delta_prefinal,
            generated_stat_final,
            delta_final,
            final_response,
        ]
    )


def set_hf_token(token_path):
    # hugging face token
    with open(token_path, "r") as hf_token_file:
        HF_TOKEN = hf_token_file.read().strip()
    #HfFolder.save_token(HF_TOKEN)


def get_final_columns():

    # colnames for final output CSV
    final_columns = [
        "questions",
        "llama_base_output",
        "helper_output",
        "cancer_entities",
        "gene_entities",
        "mutation_entities",
        "modified_prompt",
        "ground_truth_stat",
        "llama_base_stat",
        "delta_llama",
        "final_response",
    ]
    return final_columns

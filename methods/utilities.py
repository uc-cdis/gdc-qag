#!/usr/bin/env python3
# various utility functions employed by the pipeline
import json
import re
import time

import numpy as np
import pandas as pd
import spacy
import torch
import warnings

from functools import reduce, wraps
from itertools import chain
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# spacy warning, upgrade to latest spacy
# in the next release
warnings.filterwarnings(
    "ignore",
    message="Possible set union at position",
    category=FutureWarning
)


from huggingface_hub import HfFolder, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BertForSequenceClassification


from methods import gdc_api_calls


intent_expansion = {
    'cnv_and_ssm': 'copy number variants or simple somatic mutations',
    'freq_cnv_loss_or_gain': 'copy number variant losses or gains',
    'msi_h_frequency': 'microsatellite instability',
    'freq_cnv_loss_or_gain_comb': 'copy number variant losses or gains',
    'ssm_frequency': 'simple somatic mutations',
    'top_cases_counts_by_gene': 'copy number variants or simple somatic mutations'
}


def timeit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        print(f"{fn.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper



def get_top_k_cancer_entities(query, row_embeddings, project_rows, top_k=20):
    top_cancer_entities = []
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, row_embeddings)
    # best_idx = cosine_scores.argmax() best row
    # print("Best row:", rows[best_idx])
    top_results = torch.topk(cosine_scores, k=top_k)
    top_results_indices = top_results.indices.tolist()
    top_results_scores = top_results.values.tolist()
    print(top_results_scores)
    for idx, score in enumerate(top_results_scores[0]):
        if score > 0.5:
            row_idx = top_results_indices[0][idx]
            print('best row, score: {} {}'.format(project_rows[row_idx], score))
            top_cancer_entities.append([project_rows[row_idx], score])
    try:
        top_projects = [sublist[0].split(',')[0] for sublist in top_cancer_entities]
    except Exception as e:
        top_projects = []
    return top_projects



def load_llama_llm(AUTH_TOKEN):
    # hugging face model
    # https://huggingface.co/blog/llama32
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, 
        token=AUTH_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            token=AUTH_TOKEN
    )
    model = model.to('cuda')
    model = model.eval()

    return model, tok


def load_gdc_genes_mutations_hf(AUTH_TOKEN):
    dataset_id = 'uc-ctds/GDC-QAG-genes-mutations'
    filename = 'gdc_genes_mutations.json'
    json_path = hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
        token=AUTH_TOKEN
    )
    with open(json_path, 'r') as f:
        gdc_genes_mutations = json.load(f)
    return gdc_genes_mutations



def load_intent_model_hf(AUTH_TOKEN):
    model_id = 'uc-ctds/query_intent'
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True,
        token=AUTH_TOKEN
    )
    model = BertForSequenceClassification.from_pretrained(
        model_id)
    return model, tok



def infer_user_intent(query, intent_model, intent_tok):
    # model, tokenizer = load_intent_model(intent_model_path)
    intent_labels = {
        "ssm_frequency": 0.0,
        "msi_h_frequency": 1.0,
        "freq_cnv_loss_or_gain": 2.0,
        "top_cases_counts_by_gene": 3.0,
        "cnv_and_ssm": 4.0,
    }
    # set device and load both model and query on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intent_model.to(device)
    inputs = intent_tok(query, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # pass tokenized input through the model
    outputs = intent_model(**inputs)
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



def calculate_ssm_frequency(ssm_statistics, total_case_count, cancer_entities, project_mappings):
    ssm_frequency = {}
    for project in ssm_statistics.keys():
        freq = (
            ssm_statistics[project]["ssm_counts"]
            / total_case_count[project]
        )
        ssm_frequency[project] = {"frequency": round(freq * 100, 2)}
    
    # if there are no ssms, set to 0 counts
    for c in cancer_entities:
        if c not in ssm_frequency:
            ssm_frequency[c] = {'frequency': 0.0}

    print('ssm_frequency {}'.format(ssm_frequency))
    return ssm_frequency


def calculate_joint_ssm_frequency(ssm_statistics, total_case_count, mutation_list, cancer_entities):
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
        print('getting shared cases...')
        shared_cases = list(reduce(lambda x, y: x & y, cases_with_mutation))
        print('number of shared cases: {}'.format(len(shared_cases)))
        if shared_cases:
            if project not in joint_ssm_frequency:
                joint_ssm_frequency[project] = {}
            joint_frequency = len(shared_cases) / total_case_count[project]
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
    print('preparing a GDC Result for query augmentation...')
    if result_type == "joint_frequency":
        for k, v in result.items():
            if k == "joint_frequency":
                for k2, v2 in v.items():
                    gdc_result = "joint frequency in {} is {}%".format(k2, v2["joint_frequency"])
                    result_text.append(gdc_result)
    else:
        for k, v in result.items():
            if k != "joint_frequency":
                for k2, v2 in v.items():
                    gdc_result = "The frequency of {} in {} is {}%".format(k, k2, v2["frequency"])
                    result_text.append(gdc_result)
    print('prepared GDC Result: {}'.format(gdc_result))
    return result_text


def get_ssm_frequency(
    gene_entities, mutation_entities, cancer_entities, project_mappings
):
    total_case_count = {}
    mutation_list = []
    result = {}
    ssm_statistics = {}

    for ce in cancer_entities:
        total_case_count[ce] = gdc_api_calls.get_available_ssm_data_for_project(ce)

    # to match the genes with mutations
    if len(mutation_entities) > len(gene_entities):
        gene_entities = gene_entities * len(mutation_entities) 
    for gene, mutation in zip(gene_entities, mutation_entities):
        mutation_name = "_".join([gene, mutation])
        mutation_list.append(mutation_name)
        ssm_id = gdc_api_calls.get_ssm_id(gene, mutation)
        ssm_counts_by_project = gdc_api_calls.get_ssm_counts(ssm_id, cancer_entities)        
        ssm_statistics[mutation_name] = ssm_counts_by_project
        result[mutation_name] = calculate_ssm_frequency(
            ssm_statistics[mutation_name], total_case_count, cancer_entities, project_mappings
        )
    
    print('\nStep 5: Query GDC and process results\n')
    for mut in mutation_list:
        print('mutation: {}'.format(mut))
        for ce in cancer_entities:
            try:
                print('number of cases with mutation: {}'.format(
                    ssm_statistics[mut][ce]["ssm_counts"]))
            except Exception as e:
                print('number of cases with mutation: {}'.format(
                    ssm_statistics
                ))
            print('total case count: {}'.format(
                total_case_count[ce]))


    # only supporting for two mutations atm
    if len(mutation_list) > 1:
        # print('computing joint frequency')
        result["joint_frequency"] = calculate_joint_ssm_frequency(
            ssm_statistics, total_case_count, mutation_list, cancer_entities
        )
        result_text = flatten_ssm_results_to_text(result, result_type="joint_frequency")
    else:
        result["joint_frequency"] = 0
        result_text = flatten_ssm_results_to_text(
            result, result_type="single_frequency"
        )
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
    project_rows, row_embeddings = gdc_api_calls.get_project_embeddings()
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
            # try embedding based match
            if not final_entities:
                print('test embedding based match')
                for i in initial_cancer_entities:
                    c_entities = get_top_k_cancer_entities(i, row_embeddings, project_rows)
                    final_entities.append(c_entities)
                final_entities = list(chain.from_iterable(final_entities))
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



def postprocess_llm_description(tok, descriptive_response):
    try:
        num_tokens = len(tok.encode(descriptive_response))
        if num_tokens < 100:
            postprocessed_desc_response = descriptive_response
        else:
            response_list = re.split(r'\.(?!\d+%)', descriptive_response)
            # remove empty elements
            filtered_list = list(filter(None, response_list))
            postprocessed_desc_response = '.'.join(filtered_list[:-1])
    except Exception as e:
        print('unable to postprocess LLM gene description {}'.format(
            str(e)
        ))
        postprocessed_desc_response = 'unable to postprocess LLM gene description'

    if not postprocessed_desc_response.endswith('.'):
        postprocessed_desc_response += '.'
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
    return gdc_qag_base_stat, final_gdc_qag_percentage_response


@timeit
def postprocess_response(tok, row):
    # three goals:
    # goal 1:
    # check/confirm the results in gdc-qag percentage response
    # return a percentage response for gdc-qag
    # goal 2:
    # postprocess descriptive response
    # goal 3:
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
    # extract gdc_result percentage from gdc_result
        match = re.search(pattern, gdc_result)
        if match:
            gdc_result_percentage = float(match.group(1))
        else:
            gdc_result_percentage = np.nan
            print('no data available in gdc')    
    except Exception as e:
        print('unable to extract percentage from gdc result {}'.format(
            str(e)))
        gdc_result_percentage = np.nan


    try:
        # this is the LLM generated response with freq, after seeing gdc_result
        gdc_qag_percentage_response = row['percentage_response']
    except Exception as e:
        print('LLM generated gdc_qag percentage response not found, returning nan {}'.format(
            str(e)
        ))
        gdc_qag_percentage_response = np.nan
    
    try:
        # extract gdc_qag percentage from LLM response
        gdc_qag_base_stat = float(re.search(pattern, gdc_qag_percentage_response).group(1))
    except Exception as e:
        print('unable to extract percentage from gdc_qag percentage response {}'.format(
            str(e)))
        gdc_qag_base_stat = np.nan
    

    # llama-3B base output
    llama_base_output = row["llama_base_output"]

    try:
        # extract llama percentage from llama base output
        llama_base_stat = float(re.search(pattern, llama_base_output).group(1))
    except Exception as e:
        print('unable to extract llama base stat {}'.format(str(e)))
        llama_base_stat = np.nan
    
    
    ############ postprocess LLM description + percentage ###############

    final_gdc_qag_desc_response = postprocess_llm_description(
        tok, row['descriptive_response']
    )

    gdc_qag_base_stat, final_gdc_qag_percentage_response = postprocess_percentage_response(
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



def set_hf_token(token_path):
    # hugging face token
    with open(token_path, "r") as hf_token_file:
        HF_TOKEN = hf_token_file.read().strip()
    HfFolder.save_token(HF_TOKEN)



def get_final_columns():

    # colnames for final output CSV
    final_columns = [
        "questions",
        "gene_entities",
        "mutation_entities",
        "cancer_entities",
        "intent",
        "llama_base_output",
        "llama_base_stat",
        "gdc_result",
        "gdc_qag_base_stat",
        "descriptive_prompt",
        "percentage_prompt",
        "final_gdc_qag_response",
    ]
    return final_columns



#!/usr/bin/env python3
import ast
import glob
import json
import os
from functools import reduce
from pathlib import Path

import pandas as pd
import requests

proj_root = Path(__file__).resolve().parent.parent


# match "lymphoid leukemia" in query to "lymphoid leukemias" in GDC disease_type
# load project_mappings
# the function to create this tsv file is a one-time run, found as one of the api functions below
project_mappings = pd.read_csv(
    os.path.join(proj_root, "csvs", "gdc_projects.tsv"), 
    sep="\t", index_col=0, names=["project", "desc"]
)
project_mappings["desc"] = project_mappings["desc"].apply(ast.literal_eval)
project_mappings = project_mappings["desc"].to_dict()


def get_gene_mutation_data(start, stop, step):
    # cannot query the entire thing at once, need to do it in parts
    for mini_stop in range(start, stop, step):
        if mini_stop != 0:
            # curl_cmd = "https://api.gdc.cancer.gov/ssms?fields=gene_aa_change&from={}&size={}".format(start, mini_stop)
            # print('curl cmd {}'.format(curl_cmd))
            response = requests.get(curl_cmd)
            out_file = "_".join([str(start), str(mini_stop), "gene.mutation.txt"])
            with open(out_file, "w") as response_out:
                response_out.write(response.text)
            start = mini_stop
    # final curl_cmd
    curl_cmd = (
        "https://api.gdc.cancer.gov/ssms?fields=gene_aa_change&from={}&size={}".format(
            start, stop
        )
    )
    # print('curl cmd {}'.format(curl_cmd))
    response = requests.get(curl_cmd)
    out_file = "_".join([str(start), str(stop), "gene.mutation.txt"])
    with open(out_file, "w") as response_out:
        response_out.write(response.text)


def process_gene_mutation_data():
    gdc_genes = {}
    gene_mutation_data_files = glob.glob("*gene.mutation.txt")
    # print('gene_mutation_data_files {}'.format(gene_mutation_data_files))
    for f in gene_mutation_data_files:
        # print('processing file {}'.format(f))
        with open(f, "r") as f_in:
            data = json.load(f_in)
            for item in data["data"]["hits"]:
                for gene_aa_change in item["gene_aa_change"]:
                    gene, mutation = gene_aa_change.split(" ")
                    if not gene in gdc_genes:
                        gdc_genes[gene] = []
                    if not mutation in gdc_genes[gene]:
                        gdc_genes[gene].append(mutation)

    with open("gdc_genes_mutations.json", "w") as f_out:
        json.dump(gdc_genes, f_out, indent=4)


# this function creates the project mappings tsv file
# only to be run once
def get_gdc_project_ids(start, stop):
    project_mappings = {}
    curl_cmd = "https://api.gdc.cancer.gov/projects?fields=project_id,disease_type,primary_site,name&from={}&size={}".format(
        start, stop
    )
    # print('curl cmd {}'.format(curl_cmd))
    out_file = "gdc_projects.tsv"
    try:
        response = requests.get(curl_cmd)
        # print('status code {}'.format(response.status_code))
        with open(out_file, "w") as response_out:
            for item in response.json()["data"]["hits"]:
                disease_type_and_name = item["disease_type"] + [item["name"]]
                line = f"{item['project_id']}\t{disease_type_and_name}\n"
                response_out.write(line)
                project_mappings[item["project_id"]] = disease_type_and_name
        # print('project_mappings {}'.format(project_mappings))
    except Exception as e:
        print("unable to execute GDC API request {}".format(str(e)))
    return project_mappings


def get_ssm_id(gene, mutation):
    ssm_id_endpt = "https://api.gdc.cancer.gov/ssms"
    fields = ["mutation_type"]
    fields = ",".join(fields)
    expand = ["consequence.transcript"]
    filters = {
        "op": "=",
        "content": {"field": "ssms.gene_aa_change", "value": "[gene][mutation]"},
    }
    filters["content"]["value"] = gene + " " + mutation
    # print('filters {}'.format(filters))
    params = {
        "filters": json.dumps(filters),
        "fields": fields,
        "expand": expand,
        "size": 10,
    }
    try:
        print('build API call, endpt: {}'.format(ssm_id_endpt))
        print('params: {}'.format(params))
        response = requests.get(ssm_id_endpt, params=params)
        response_json = json.loads(response.content)
        ssm_id = response_json["data"]["hits"][0]["id"]
    except Exception as e:
        print('obtained ssm id {}'.format(ssm_id))
        print("unable to execute GDC API request {}".format(str(e)))
        ssm_id = None
    return ssm_id


def get_ssm_counts(ssm_id, cancer_entities):
    # get project level counts of ssm
    ssm_counts_by_project = {}

    for ce in cancer_entities:
        ssm_occurrences_endpt = "https://api.gdc.cancer.gov/ssm_occurrences"
        fields = ["case.project.project_id", "case.case_id"]
        fields = ",".join(fields)
        filters = {
            "op": "and", 
            "content": [
                {   
                    "op": '=',
                    "content": {"field": "ssm.ssm_id", "value": ssm_id}
                },
                {
                    "op": "=", 
                    "content": {"field": "case.project.project_id", "value": ce}
                },
            ]}
        params = {"filters": json.dumps(filters), "fields": fields, "size": 1000}
        try:
            print('build API call, endpt: {}'.format(ssm_occurrences_endpt))
            print('params: {}'.format(params))
            response = requests.get(ssm_occurrences_endpt, params=params)
            ssm_counts = json.loads(response.content)
            for item in ssm_counts["data"]["hits"]:
                project_name = item["case"]["project"]["project_id"]
                case_id_list = "case_id_list"
                if not project_name in ssm_counts_by_project:
                    ssm_counts_by_project[project_name] = {}
                    ssm_counts_by_project[project_name][case_id_list] = []
                ssm_counts_by_project[project_name][case_id_list].append(
                    item["case"]["case_id"]
                )
                ssm_counts_by_project[project_name]["ssm_counts"] = (
                    ssm_counts_by_project[project_name]["ssm_counts"] + 1
                    if "ssm_counts" in ssm_counts_by_project[project_name]
                    else 1
                )
        except Exception as e:
            print("unable to execute GDC API request {}".format(str(e)))
    print('ssm counts by proj {}'.format(ssm_counts_by_project))
    return ssm_counts_by_project


def get_available_cnv_data_for_project(project):
    case_ssm_endpt = "https://api.gdc.cancer.gov/case_ssms"
    fields = ["project.project_id", "available_variation_data"]
    fields = ",".join(fields)
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {"field": "available_variation_data", "value": "cnv"},
            },
            {"op": "=", "content": {"field": "project.project_id", "value": project}},
        ],
    }
    params = {"filters": json.dumps(filters), "fields": fields, "size": 1000}
    try:
        print('build API call, endpt: {}'.format(case_ssm_endpt))
        print('params: {}'.format(params))
        response = requests.get(case_ssm_endpt, params=params)
        response_json = json.loads(response.content)
        total_case_count = response_json["data"]["pagination"]["total"]
    except Exception as e:
        print("unable to execute GDC API request {}".format(str(e)))
        total_case_count = 0
    # print('total case count {}'.format(total_case_count))
    return total_case_count


def get_available_ssm_data_for_project(project):
    case_ssm_endpt = "https://api.gdc.cancer.gov/case_ssms"
    fields = ["project.project_id", "available_variation_data"]
    fields = ",".join(fields)

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {"field": "available_variation_data", "value": "ssm"},
            },
            {"op": "=", "content": {"field": "project.project_id", "value": project}},
        ],
    }
    params = {"filters": json.dumps(filters), "fields": fields, "size": 1000}
    try:
        print('build API call, endpt: {}'.format(case_ssm_endpt))
        print('params: {}'.format(params))
        response = requests.get(case_ssm_endpt, params=params)
        response_json = json.loads(response.content)
        total_case_count = response_json["data"]["pagination"]["total"]
    except Exception as e:
        print("unable to execute GDC API request {}".format(str(e)))
    return total_case_count


def get_top_mutated_genes_by_project(cancer_entities, top_k):
    # need an AI way of recognizing top k from query, here using 10 as default
    top_mutated_genes_by_project = {}
    # if cancer_entities is empty, initialize some entities
    if not cancer_entities:
        cancer_entities = list(project_mappings.keys())

    for ce in cancer_entities:
        endpt = "https://api.gdc.cancer.gov/analysis/top_mutated_genes_by_project"

        fields = ["gene_id", "symbol"]
        fields = ",".join(fields)

        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {"field": "case.project.project_id", "value": [ce]},
                }
            ],
        }
        params = {"filters": json.dumps(filters), "fields": fields, "size": 1000}
        try:
            print('build API call, endpt: {}'.format(endpt))
            print('params: {}'.format(params))
            response = requests.get(endpt, params=params)
            response_json = json.loads(response.content)
            top_mutated_genes_by_project[ce] = response_json["data"]["hits"][:top_k]
        except Exception as e:
            print("unable to execute GDC API request {}".format(str(e)))
    return top_mutated_genes_by_project


def return_joint_single_cnv_frequency(cnv, cnv_change, cnv_change_5_category):
    result_text = []
    # set category for heterozygous del
    if not cnv_change_5_category:
        if cnv_change == "Loss":
            cnv_change_5_category = "Heterozygous Deletion"
        # print('formatting results {}'.format(cnv_change_5_category))
    cnv_freq = {}
    for ce, v in cnv.items():
        cnv_freq[ce] = {}
        genes = list(v.keys())
        # print('ce, genes {} {}'.format(ce, genes))
        total_number_of_cases_with_cnv_data = get_available_cnv_data_for_project(ce)
        # skip if total number of cnv cases from API is 0
        if not total_number_of_cases_with_cnv_data:
            print('could not retrieve total number of cases with CNV data for {}'.format(ce))
            total_number_of_cases_with_cnv_data = 0


        print('\nStep 5: Query GDC and process results\n')
        print('total number of cases with CNV data {}'.format(
                total_number_of_cases_with_cnv_data))

        if len(genes) > 1:
            cases_with_cnvs = [set(cnv[ce][g]["case_id_list"]) for g in genes]
            print('genes: {}'.format(genes))
            num_cases_with_cnvs = [len(i) for i in cases_with_cnvs]
            print('number of cases with CNVs: {}'.format(num_cases_with_cnvs))
            print('getting shared cases...')
            shared_cases = list(reduce(lambda x, y: x & y, cases_with_cnvs))
            print('number of shared cases {}'.format(len(shared_cases)))
            print('preparing a GDC Result for query augmentation...')
            try:
                joint_frequency = round(
                    (len(shared_cases) / total_number_of_cases_with_cnv_data) * 100, 2
                )
            except Exception as e:
                joint_frequency = 0
            gdc_result = "joint frequency in {} is {}%".format(ce, joint_frequency)
            print('prepared GDC Result: {}'.format(gdc_result))
            result_text.append(gdc_result)
        else:
            joint_frequency = 0
            num_cases_with_cnvs = len(set(cnv[ce][genes[0]]["case_id_list"]))
            print('number of cases with cnvs {}'.format(num_cases_with_cnvs))
            try:
                frequency = round((num_cases_with_cnvs / total_number_of_cases_with_cnv_data) * 100, 2)
            except Exception as e:
                frequency = 0
            
            for k2, v2 in v.items():
                print('preparing a GDC Result for query augmentation...')
                gdc_result = "The frequency of {} {} in {} is {}%".format(
                        k2, cnv_change_5_category, ce, frequency
                    )
                print('prepared GDC Result: {}'.format(gdc_result))
                result_text.append(gdc_result)
    return result_text


def get_cnv_filter_with_cnv_change_category(cnv_change, ce, ge, cnv_change_5_category):

    filter = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cnv.cnv_change", "value": [cnv_change]}},
            {
                "op": "in",
                "content": {
                    "field": "cnv.cnv_change_5_category",
                    "value": [cnv_change_5_category],
                },
            },
            {
                "op": "=",
                "content": {"field": "cnv.consequence.gene.symbol", "value": ge},
            },
            {"op": "=", "content": {"field": "case.project.project_id", "value": ce}},
        ],
    }
    return filter


def get_freq_cnv_loss_or_gain(gene_entities, cancer_entities, query, cnv_and_ssm_flag):
    cnv = {}
    lc_query = query.lower()
    # need to figure out how to get deletion and gain
    # V1 is only co-deletion, or co-gain
    loss_terms = ["loss", "loh", "deletion", "co-deletion", "lost", "LOH"]
    if any(term in lc_query for term in loss_terms):
        cnv_change = "Loss"
        if "homozygous" in lc_query:
            cnv_change_5_category = "Homozygous Deletion"
        else:
            cnv_change_5_category = "Loss"
    else:
        cnv_change = "Gain"
        if "amplification" in lc_query:
            cnv_change_5_category = "Amplification"
        else:
            cnv_change_5_category = "Gain"

    if not cancer_entities:
        cancer_entities = list(project_mappings.keys())

    # print('cnv change, cnv change 5 category in query {} {}'.format(
    #  cnv_change, cnv_change_5_category))

    for ce in cancer_entities:
        for ge in gene_entities:
            # print('processing {}, {}'.format(ce, ge))
            endpt = "https://api.gdc.cancer.gov/cnv_occurrences"
            fields = [
                "cnv.chromosome",
                "cnv.cnv_change",
                "cnv.cnv_change_5_category" "cnv.consequence.gene.symbol",
                "case.case_id",
                "case.project.project_id",
            ]
            fields = ",".join(fields)
            filters = get_cnv_filter_with_cnv_change_category(
                cnv_change, ce, ge, cnv_change_5_category
            )
            params = {"filters": json.dumps(filters), "fields": fields, "size": 1000}
            try:
                # print('filters {}'.format(json.dumps(filters)))
                # skip if response not successful
                print('build API call, endpt: {}'.format(endpt))
                print('params: {}'.format(params))
                response = requests.get(endpt, params=params)
                response_json = json.loads(response.content)
            except Exception as e:
                print("exception: {}".format(str(e)))
                continue

            if not ce in cnv:
                cnv[ce] = {}
            if not ge in cnv[ce]:
                cnv[ce][ge] = {}

            case_id_list = []
            for item in response_json["data"]["hits"]:
                if item["case"]["case_id"]:
                    case_id_list.append(item["case"]["case_id"])
            number_of_cases_with_cnv_change = len(case_id_list)
            cnv[ce][ge]["case_id_list"] = case_id_list

    # print('debug: cnv {}'.format(cnv))
    if cnv_and_ssm_flag:
        return cnv
    else:
        result_text = return_joint_single_cnv_frequency(
            cnv, cnv_change, cnv_change_5_category
        )
        cancer_entities = list(cnv.keys())
        return result_text, cancer_entities


def get_msi_frequency(cancer_entities):
    msi_h_frequency = {}
    result_text = []
    # init some starting cancer entities if none
    if not cancer_entities:
        cancer_entities = list(project_mappings.keys())
    for ce in cancer_entities:
        endpt = "https://api.gdc.cancer.gov/files"
        fields = [
            "cases.project.project_id",
            "msi_score",
            "msi_status",
            "experimental_strategy",
        ]
        fields = ",".join(fields)

        filters = {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "data_format", "value": "BAM"}},
                {
                    "op": "in",
                    "content": {
                        "field": "experimental_strategy",
                        "value": ["WXS", "WGS"],
                    },
                },
                {
                    "op": "in",
                    "content": {"field": "cases.project.project_id", "value": [ce]},
                },
            ],
        }
        params = {"filters": json.dumps(filters), "fields": fields, "size": 10000}
        try:
            print('build API call, endpt: {}'.format(endpt))
            print('params: {}'.format(params))
            response = requests.get(endpt, params=params)
            response_json = json.loads(response.content)

            msi_results = []
            for item in response_json["data"]["hits"]:
                # only score tumors where MSI status is computed for frequency
                if "msi_status" in item:
                    # exclude None
                    if item['msi_status']:
                        msi_results.append(item["msi_status"])
            msi_pos = msi_results.count('MSI')
            msi_total = len(msi_results)
            freq = msi_pos / msi_total
            print('\nStep 5: Query GDC and process results\n')
            print('obtained {} BAM files with MSI tag, out of a total of {} BAM files with MSI information'.format(
                msi_pos, msi_total
            ))
            msi_h_frequency[ce] = {"frequency": round(freq * 100, 2)}
            print('preparing a GDC Result for query augmentation...')
            gdc_result = "The frequency of MSI in {} is {}%".format(
                    ce, msi_h_frequency[ce]["frequency"]
                )
            print('prepared GDC Result: {}'.format(gdc_result)) 
            result_text.append(gdc_result)
        except Exception as e:
            print("unable to execute GDC API request {}".format(str(e)))
    ce_api_success = list(msi_h_frequency.keys())
    return result_text, ce_api_success


def get_ensembl_gene_ids(gene_entities):
    ensembl_gene_ids = []
    for ge in gene_entities:
        endpt = "https://api.gdc.cancer.gov/genes"
        fields = ["gene_id"]
        fields = ",".join(fields)
        filters = {
            "op": "and",
            "content": [{"op": "=", "content": {"field": "symbol", "value": ge}}],
        }
        params = {"filters": json.dumps(filters), "fields": fields, "size": 100}
        try:
            print('build API call, endpt: {}'.format(endpt))
            print('params: {}'.format(params))
            response = requests.get(endpt, params=params)
            response_json = json.loads(response.content)
            ensembl_gene_ids.append(response_json["data"]["hits"][0]["gene_id"])
        except Exception as e:
            print("unable to execute GDC API request {}".format(str(e)))
    return ensembl_gene_ids


def get_total_variation_data_for_project(project):
    case_ssm_endpt = "https://api.gdc.cancer.gov/case_ssms"
    fields = ["project.project_id", "available_variation_data"]
    fields = ",".join(fields)

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "available_variation_data",
                    "value": ["ssm", "cnv"],
                },
            },
            {"op": "=", "content": {"field": "project.project_id", "value": project}},
        ],
    }
    params = {"filters": json.dumps(filters), "fields": fields, "size": 1000}
    try:
        print('build API call, endpt: {}'.format(case_ssm_endpt))
        print('params: {}'.format(params))
        response = requests.get(case_ssm_endpt, params=params)
        response_json = json.loads(response.content)
        total_case_count = response_json["data"]["pagination"]["total"]
    except Exception as e:
        print("unable to execute GDC API request {}".format(str(e)))
        total_case_count = 0

    return total_case_count


def get_cases_with_ssms_in_a_gene(project, gene_name):

    result = {}
    endpt = "https://api.gdc.cancer.gov/ssm_occurrences"
    fields = ["case.case_id"]
    fields = ",".join(fields)

    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {"field": "case.project.project_id", "value": project},
            },
            {
                "op": "in",
                "content": {
                    "field": "ssm.consequence.transcript.gene.symbol",
                    "value": gene_name,
                },
            },
        ],
    }
    params = {"filters": json.dumps(filters), "fields": fields, "size": 1000}
    try:
        print('build API call, endpt: {}'.format(endpt))
        print('params: {}'.format(params))
        response = requests.get(endpt, params=params)
        response_json = json.loads(response.content)
        case_id_list = []
        for item in response_json["data"]["hits"]:
            if item["case"]["case_id"]:
                case_id_list.append(item["case"]["case_id"])
        result["case_id_list"] = list(set(case_id_list))
    except Exception as e:
        print("unable to execute GDC API request {}".format(str(e)))
    return result


def run_cnv_ssm_api(decompose_result, cancer_entities, query):
    """
    decompose_result['cnv_and_ssm'] = True
    decompose_result['cnv_gene'] = cnv_gene.split(':')[1]
    decompose_result['mut_gene'] = mut_gene.split(':')[1]
    decompose_result['cnv_change_type'] = match_term
    """
    gene_entities = []
    cases_with_ssm_and_cnvs = []
    result = []
    gene_entities.append(decompose_result["cnv_gene"])
    cnv_result = get_freq_cnv_loss_or_gain(
        gene_entities, cancer_entities, query, cnv_and_ssm_flag=True
    )

    for ce in cancer_entities:

        try:
            # get_cases_with_ssms_in_a_gene returns the number of cases with ssms
            ssm_result = get_cases_with_ssms_in_a_gene(
                project=ce, gene_name=decompose_result["mut_gene"]
            )
            total_case_count = get_total_variation_data_for_project(project=ce)
            print('\nStep 5: Query GDC and process results\n')
            # calcuate overlap of cases and return freq
            print('getting shared cases with CNV and SSMs...')
            cases_with_ssm_and_cnvs = [
                set(cnv_result[ce][decompose_result["cnv_gene"]]["case_id_list"]),
                set(ssm_result["case_id_list"]),
            ]
            shared_cases = list(reduce(lambda x, y: x & y, cases_with_ssm_and_cnvs))
            print('number of shared_cases {}'.format(len(shared_cases)))
            print('total case count {}'.format(total_case_count))
            freq = round((len(shared_cases) / total_case_count) * 100, 2)
            print('preparing a GDC Result for query augmentation...')
            gdc_result = "The joint frequency in {} is {}%".format(ce, freq)
        except Exception as e:
            gdc_result = "joint freq in {} is not available".format(ce)
        print('prepared GDC Result {}'.format(gdc_result))
        result.append(gdc_result)
    return result, cancer_entities


def get_top_cases_counts_by_gene(gene_entities, cancer_entities):
    top_cases_counts_by_gene = {}
    result = []
    emsembl_gene_ids = get_ensembl_gene_ids(gene_entities)
    if not cancer_entities:
        cancer_entities = list(project_mappings.keys())
    for ce in cancer_entities:
        top_cases_counts_by_gene[ce] = {}
        # note this gives you ssm + cnv
        endpt = "https://api.gdc.cancer.gov/analysis/top_cases_counts_by_genes?gene_ids={}".format(
            ",".join(emsembl_gene_ids)
        )
        print('build API call, endpt: {}'.format(endpt))
        response = requests.get(endpt)
        response_json = json.loads(response.content)
        try:
            for item in response_json["aggregations"]["projects"]["buckets"]:
                if item["key"] == ce:
                    cases_with_mutations = item["doc_count"]
            # total_case_count = get_available_ssm_data_for_project(ce)
            total_case_count = get_total_variation_data_for_project(project=ce)
            cases_without_mutations = total_case_count - cases_with_mutations
            top_cases_counts_by_gene[ce]["cases_with_mutations"] = cases_with_mutations
            top_cases_counts_by_gene[ce][
                "cases_without_mutations"
            ] = cases_without_mutations
            top_cases_counts_by_gene[ce]["total_case_count"] = total_case_count
            print('\nStep 5: Query GDC and process results\n')
            print('obtained {} cases with mutations and a total case count of {}'.format(
                cases_with_mutations, total_case_count
            ))
            freq = cases_with_mutations / total_case_count
            top_cases_counts_by_gene[ce]["frequency"] = round(freq * 100, 2)
            print('preparing a GDC Result for query augmentation...')
            gdc_result = "The frequency of cases with mutations in {} is {}%".format(
                    ce, top_cases_counts_by_gene[ce]["frequency"]
                )
            result.append(gdc_result)
        except Exception as e:
            result.append("frequency unavailable from API for {}".format(ce))
    print('prepared GDC Result {}'.format(gdc_result))
    cancer_entities = list(top_cases_counts_by_gene.keys())
    return result, cancer_entities


def get_project_summary(cancer_entities):
    project_summary = {}
    for ce in cancer_entities:
        endpt = "https://api.gdc.cancer.gov/projects/{}?expand=summary,summary.experimental_strategies,summary.data_categories".format(
            ce
        )
        response = requests.get(endpt)
        response_json = json.loads(response.content)
        project_summary[ce]["project_summary"] = response_json["data"]
    return project_summary


def map_cancer_entities_to_project(initial_cancer_entities, project_mappings):
    project_match = {}
    for ce in initial_cancer_entities:
        # cancer_wild_card = '*' + ce
        endpoint = "https://api.gdc.cancer.gov/projects"
        fields = ["project_id", "disease_type", "name"]
        fields = ",".join(fields)

        filters = {"op": "=", "content": {"field": "name", "value": [ce]}}
        params = {"filters": json.dumps(filters), "fields": fields, "size": 10000}
        try:
            response = requests.get(endpoint, params=params)
            response_json = json.loads(response.content)
            # print('response_json {}'.format(json.dumps(
            #    response_json, indent=4)))
            for item in response_json["data"]["hits"]:
                project_id = item["project_id"]
            project_match[ce] = project_id
        except Exception as e:
            pass
            # print('unable to return a match from projects endpt '
            #      'perform further checks on project_mappings')
    return project_match

#!/usr/bin/env python3

from . import gdc_api_calls as gdc_api
from . import utilities as util


def get_ssm_frequency(
    intent: str,
    gene_entities: list,
    mutation_entities: list,
    cancer_entities: list,
    project_mappings: dict,
):
    print(f"getting results for {intent}")
    result, cancer_entities = util.get_ssm_frequency(
        gene_entities, mutation_entities, cancer_entities, project_mappings
    )
    return result, cancer_entities


def get_freq_cnv_loss_or_gain(
    intent: str, gene_entities: list, cancer_entities: list, query: str
):
    print(f"getting results for {intent}")
    result, cancer_entities = gdc_api.get_freq_cnv_loss_or_gain(
        gene_entities, cancer_entities, query, cnv_and_ssm_flag=False
    )
    return result, cancer_entities


def get_msi_h_frequency(intent: str, cancer_entities: list):
    print(f"getting results for {intent}")
    result, cancer_entities = gdc_api.get_msi_frequency(cancer_entities)
    return result, cancer_entities


def get_cnv_and_ssm(
    intent: str,
    query: str,
    cancer_entities: list,
    gene_entities: list,
    gdc_genes_mutations: dict,
):
    print(f"getting results for {intent}")
    result, cancer_entities = util.get_freq_of_cnv_and_ssms(
        query, cancer_entities, gene_entities, gdc_genes_mutations
    )
    return result, cancer_entities


def get_top_cases_counts_by_gene(
    intent: str, cancer_entities: list, gene_entities: list
):
    print(f"getting results for {intent}")
    result, cancer_entities = gdc_api.get_top_cases_counts_by_gene(
        gene_entities, cancer_entities
    )
    return result, cancer_entities


# define list of functions
tool_functions_list = {
    "get_ssm_frequency": get_ssm_frequency,
    "get_freq_cnv_loss_or_gain": get_freq_cnv_loss_or_gain,
    "get_msi_h_frequency": get_msi_h_frequency,
    "get_cnv_and_ssm": get_cnv_and_ssm,
    "get_top_cases_counts_by_gene": get_top_cases_counts_by_gene,
}


# define individual tools
def construct_tools_list(
    gene_entities,
    mutation_entities,
    cancer_entities,
    project_mappings,
    query,
    gdc_genes_mutations,
):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_ssm_frequency",
                "description": "Get results for ssm_frequency intent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "intent to guide endpoint mapping, e.g., 'ssm_frequency'",
                        },
                        "gene_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": gene_entities,
                        },
                        "mutation_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": mutation_entities,
                        },
                        "cancer_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": cancer_entities,
                        },
                        "project_mappings": {
                            "type": "dict",
                            "description": "optional argument",
                            "default": project_mappings,
                        },
                    },
                    "required": [
                        "intent",
                        "gene_entities",
                        "mutation_entities",
                        "cancer_entities",
                        "project_mappings",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_freq_cnv_loss_or_gain",
                "description": "Get results for freq_cnv_loss_or_gain intent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "intent to guide endpoint mapping, e.g., 'freq_cnv_loss_or_gain'",
                        },
                        "gene_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": gene_entities,
                        },
                        "cancer_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": cancer_entities,
                        },
                        "query": {
                            "type": "str",
                            "description": "optional argument",
                            "default": query,
                        },
                    },
                    "required": ["intent", "gene_entities", "cancer_entities", "query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_msi_h_frequency",
                "description": "Get results for msi_h_frequency intent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "intent to guide endpoint mapping, e.g., 'msi_h_frequency'",
                        },
                        "cancer_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": cancer_entities,
                        },
                    },
                    "required": ["intent", "cancer_entities"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_cnv_and_ssm",
                "description": "Get results for cnv_and_ssm intent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "intent to guide endpoint mapping, e.g., 'cnv_and_ssm'",
                        },
                        "query": {
                            "type": "string",
                            "description": "optional argument",
                            "default": query,
                        },
                        "gene_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": gene_entities,
                        },
                        "cancer_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": cancer_entities,
                        },
                        "gdc_genes_mutations": {
                            "type": "dict",
                            "description": "optional argument",
                            "default": gdc_genes_mutations,
                        },
                    },
                    "required": [
                        "intent",
                        "query",
                        "gene_entities",
                        "cancer_entities",
                        "gdc_genes_mutations",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_top_cases_counts_by_gene",
                "description": "Get results for top_cases_counts_by_gene intent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "intent to guide endpoint mapping e.g., 'top_cases_counts_by_gene'",
                        },
                        "gene_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": gene_entities,
                        },
                        "cancer_entities": {
                            "type": "list",
                            "description": "optional argument",
                            "default": cancer_entities,
                        },
                    },
                    "required": ["intent", "gene_entities", "cancer_entities"],
                },
            },
        },
    ]
    return tools

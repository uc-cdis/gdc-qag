import requests
import json

def percentify(num, den):
## Converts two int to float and finds frequency
    dec = float(num) / float(den)
    percent = dec * 100
    return percent


def proj_count(project, type):
## Finds number of cases that have a certain type of mutation.
## Use "cnv", "ssm", or "both"
    allowable = ["cnv","ssm"]
    if type in allowable:
        t = type
    elif type == "both":
        t = allowable
    else:
        t = ""
    case_ep = "https://api.gdc.cancer.gov/case_ssms"
    filters = {
     "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "project.project_id",
                    "value": project
                }
           },
           {
               "op": "in",
               "content": {
                   "field": "available_variation_data",
                   "value": t
               }
            }
    ]}
    params = {
     "filters": json.dumps(filters),
     "fields": "available_variation_data",
     "response": "JSON",
     "size": 2000
    }
    response = requests.get(case_ep, params=params).json()
    total = response["data"]["pagination"]["total"]
    return total


def cnv_and_ssm(gene, project):
    cnvs_endpt = "https://api.gdc.cancer.gov/cnv_occurrences"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cnv.consequence.gene.symbol",
                    "value": gene
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "case.project.project_id",
                    "value": project
                }
            }
        ]
        }

    params = {
        "filters": json.dumps(filters),
        "fields": "case.submitter_id",
        "response": "JSON",
        "size": 8000
    }
    response = requests.get(cnvs_endpt, params=params).json()
    cnv_list = []
    for hit in response["data"]["hits"]:
        case = hit["case"]["submitter_id"]
        cnv_list.append(case)


    ssm_ep = "https://api.gdc.cancer.gov/ssm_occurrences"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "case.project.project_id",
                    "value": project
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "ssm.consequence.transcript.gene.symbol",
                    "value": gene
                }
            }
            ]}

    params = {
                "filters": json.dumps(filters),
                "fields": "case.submitter_id",
                "response": "JSON",
                "size": 2000
            }
    response = requests.get(ssm_ep, params=params).json()
    ssm_list = []
    for hit in response["data"]["hits"]:
        case = hit["case"]["submitter_id"]
        ssm_list.append(case)

    union = set(cnv_list + ssm_list)
    num = len(union)
    den = proj_count(project, "both")
    percent = percentify(num, den)
    return percent


def freq_cnv_loss_or_gain(gene, project, mut):
    cnvs_endpt = "https://api.gdc.cancer.gov/cnv_occurrences"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cnv.consequence.gene.symbol",
                    "value": gene
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "case.project.project_id",
                    "value": project
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "cnv.cnv_change_5_category",
                    "value": mut
                }
            }
        ]
        }
    params = {
                "filters": json.dumps(filters),
                "fields": "case.submitter_id",
                "response": "JSON",
                "size": 2000
            }
    den = proj_count(project, "cnv")
    response = requests.get(cnvs_endpt, params=params).json()
    num = response["data"]["pagination"]["total"]
    percent = percentify(num, den)
    return percent

def freq_cnv_loss_or_gain_comb(gene1, gene2, project, mut):
    cnvs_endpt = "https://api.gdc.cancer.gov/cnv_occurrences"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cnv.consequence.gene.symbol",
                    "value": gene1
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "case.project.project_id",
                    "value": project
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "cnv.cnv_change_5_category",
                    "value": mut
                }
            }
        ]
        }
    params = {
                "filters": json.dumps(filters),
                "fields": "case.submitter_id",
                "response": "JSON",
                "size": 2000
            }
    response = requests.get(cnvs_endpt, params=params).json()
    gene1_list = []
    for hit in response["data"]["hits"]:
        case = hit["case"]["submitter_id"]
        gene1_list.append(case)

    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cnv.consequence.gene.symbol",
                    "value": gene2
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "case.project.project_id",
                    "value": project
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "cnv.cnv_change_5_category",
                    "value": mut
                }
            }
        ]
        }
    params = {
                "filters": json.dumps(filters),
                "fields": "case.submitter_id",
                "response": "JSON",
                "size": 2000
            }
    response = requests.get(cnvs_endpt, params=params).json()
    gene2_list = []
    for hit in response["data"]["hits"]:
        case = hit["case"]["submitter_id"]
        gene2_list.append(case)
    set1 = set(gene1_list)
    set2 = set(gene2_list)
    comm = set1.intersection(set2)
    num = len(comm)
    den = proj_count(project, "cnv")
    percent = percentify(num, den)
    return percent

def ssm_frequency(gene, project, mut):
    ssm_endpt = "https://api.gdc.cancer.gov/ssm_occurrences"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "ssm.consequence.transcript.gene.symbol",
                    "value": gene
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "case.project.project_id",
                    "value": project
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "ssm.consequence.transcript.aa_change",
                    "value": mut
                }
            }
        ]
        }
    params = {
                "filters": json.dumps(filters),
                "fields": "case.submitter_id",
                "response": "JSON",
                "size": 2000
            }
    den = proj_count(project, "ssm")
    response = requests.get(ssm_endpt, params=params).json()
    num = response["data"]["pagination"]["total"]
    percent = percentify(num, den)
    return percent


def msi_frequency(project):
    file_ep = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "data_format",
                    "value": "BAM"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "msi_status",
                    "value": ["msi","mss"]
                }
            }
        ]
        }
    params = {
                "filters": json.dumps(filters),
                "fields": "msi_status",
                "response": "JSON",
                "size": 2000
            }
    response = requests.get(file_ep, params=params).json()
    msi = 0
    total = 0
    for hit in response["data"]["hits"]:
        micro = hit["msi_status"]
        total +=1
        if micro == "MSI":
            msi +=1
    percent = percentify(msi, total)
    return percent


def freq_ssm_comb(gene1, gene2, project, mut1, mut2):
    ssm_endpt = "https://api.gdc.cancer.gov/ssm_occurrences"
    filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "ssm.consequence.transcript.gene.symbol",
                        "value": gene1
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "case.project.project_id",
                        "value": project
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "ssm.consequence.transcript.aa_change",
                        "value": mut1
                    }
                }
            ]
            }

    params = {
                "filters": json.dumps(filters),
                "fields": "case.submitter_id",
                "response": "JSON",
                "size": 2000
            }
    response = requests.get(ssm_endpt, params=params).json()
    gene1_list = []
    for hit in response["data"]["hits"]:
        case = hit["case"]["submitter_id"]
        gene1_list.append(case)

    filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "ssm.consequence.transcript.gene.symbol",
                        "value": gene2
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "case.project.project_id",
                        "value": project
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "ssm.consequence.transcript.aa_change",
                        "value": mut2
                    }
                }
            ]
            }

    params = {
                "filters": json.dumps(filters),
                "fields": "case.submitter_id",
                "response": "JSON",
                "size": 2000
            }
    response = requests.get(ssm_endpt, params=params).json()
    gene2_list = []
    for hit in response["data"]["hits"]:
        case = hit["case"]["submitter_id"]
        gene2_list.append(case)
    set1 = set(gene1_list)
    set2 = set(gene2_list)
    comm = set1.intersection(set2)
    num = len(comm)
    den = proj_count(project, "ssm")
    percent = percentify(num, den)
    return percent

def freq_ssmcnv_comb(ssm_gene, cnv_gene, cnv_mut, project):
    ssm_endpt = "https://api.gdc.cancer.gov/ssm_occurrences"
    cnvs_endpt = "https://api.gdc.cancer.gov/cnv_occurrences"
    filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "ssm.consequence.transcript.gene.symbol",
                        "value": ssm_gene
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "case.project.project_id",
                        "value": project
                    }
                }
            ]
            }

    params = {
                "filters": json.dumps(filters),
                "fields": "case.submitter_id",
                "response": "JSON",
                "size": 2000
            }
    response = requests.get(ssm_endpt, params=params).json()
    gene1_list = []
    for hit in response["data"]["hits"]:
        case = hit["case"]["submitter_id"]
        gene1_list.append(case)

    filters = {
        "op": "and",
        "content": [
        {
            "op": "=",
            "content": {
                "field": "cnv.consequence.gene.symbol",
                "value": cnv_gene
            }
        },
        {
            "op": "=",
            "content": {
                "field": "cnv.cnv_change_5_category",
                "value": cnv_mut
            }
        },
        {
            "op": "=",
            "content": {
                "field": "case.project.project_id",
                "value": project
            }
        }
    ]
    }

    params = {
                "filters": json.dumps(filters),
                "fields": "case.submitter_id",
                "response": "JSON",
                "size": 2000
            }
    response = requests.get(cnvs_endpt, params=params).json()
    gene2_list = []
    for hit in response["data"]["hits"]:
        case = hit["case"]["submitter_id"]
        gene2_list.append(case)
    set1 = set(gene1_list)
    set2 = set(gene2_list)
    comm = set1.intersection(set2)
    num = len(comm)
    den = proj_count(project, "both")
    percent = percentify(num, den)
    return percent


##  What is the co-occurence frequency of IDH1 R132H and TP53 R273C simple so-
##  matic mutations in the low grade glioma project TCGA-LGG in the genomic data
##  commons?

# print(freq_ssm_comb(gene1="IDH1", gene2="TP53", project="TCGA-LGG", mut1="R132H", mut2="R273C"))

##  What is the co-occurence frequency of RNF43 G659Vfs*41 and RPL22 K15Rfs*5
##  simple somatic mutations in the TCGA-UCEC project?

# print(freq_ssm_comb(gene1="RNF43", gene2="RPL22", project="TCGA-UCEC", mut1="G659Vfs*41", mut2="K15Rfs*5"))

##  What is the co-occurence frequency of ACVR2A K437Rfs*5 and RPL22 K15Rfs*5
##  simple somatic mutations in the TCGA-STAD project?

# print(freq_ssm_comb(gene1="ACVR2A", gene2="RPL22", project="TCGA-STAD", mut1="K437Rfs*5", mut2="K15Rfs*5"))

## What is the frequency of cases with amplifications in EGFR and simple somatic
## mutations in ATRX in the TCGA-LGG project? 

# print(freq_ssmcnv_comb(ssm_gene="ATRX", cnv_gene="EGFR", cnv_mut = "Amplification", project="TCGA-LGG"))

##  What is the incidence of simple somatic mutations or copy number variants
##  in PDGFRA in the genomic data commons for Uterine Carcinosarcoma TCGA-UCS
##  project?

#print(cnv_and_ssm(gene="PDGFRA", project="TCGA-UCS"))
#print(cnv_and_ssm(gene="PDGFRA", project="EXCEPTIONAL_RESPONDERS-ER"))


##  What is the frequency of somatic JAK2 heterozygous deletion in
##  Acute Lymphoblastic Leukemia - Phase II TARGET-ALL-P2 project
## in the genomic data commons?

# print(freq_cnv_loss_or_gain(gene="JAK2", project="TARGET-ALL-P2", mut="Loss"))

##  What is the co-occurence frequency of somatic heterozygous deletions in
## CDKN2A and CDKN2B in the mesothelioma project TCGA-MESO in the genomic data
## commons?

# print(freq_cnv_loss_or_gain_comb(gene1="CDKN2A", gene2="CDKN2B", project="TCGA-MESO", mut="Loss"))

##  Can you provide the frequency of NRAS gain in Skin Cutaneous Melanoma?

# print(freq_cnv_loss_or_gain(gene="NRAS", project="TCGA-SKCM", mut="Gain"))

##  In Breast Invasive Carcinoma TCGA-BRCA project data from the genomic data
##  commons, what is the frequency of ALK amplification?

# print(freq_cnv_loss_or_gain(gene="ALK", project="TCGA-BRCA", mut="Amplification"))

##  What is the incidence of somatic TP53 homozygous deletion in Osteosarcoma
##  TARGET-OS project in the genomic data commons?

# print(freq_cnv_loss_or_gain(gene="TP53", project="TARGET-OS", mut="Homozygous Deletion"))

##  What is the co-occurence frequency of somatic homozygous deletions in
##  CDKN2A and CDKN2B in the mesothelioma project TCGA-MESO in the genomic
##  data commons?

# print(freq_cnv_loss_or_gain_comb(gene1="CDKN2A", gene2="CDKN2B", project="TCGA-MESO", mut="Homozygous Deletion"))

##  What is the rate of occurrence of NRAS Q61R mutation in Skin Cutaneous
##  Melanoma TCGA-SKCM project in the genomic data commons?

# print(ssm_frequency(gene="NRAS", project="TCGA-SKCM", mut="Q61R"))

##  Can you provide the prevalence of microsatellite instability in Genomic
##  Characterization CS-MATCH-0007 Arm S1 MATCH-S1 project in the genomic data
##  commons?

# print(msi_frequency(project="MATCH-S1"))

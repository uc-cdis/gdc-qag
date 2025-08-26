#!/usr/bin/python env
# generate truth variant frequencies from the GDC for GVT 
# use questions, entities and intent from gdc-qag results for 6011 queries
# and run through this code to generate truth variant
# frequencies


import sys
from pathlib import Path

# define project root
project_root = Path().resolve().parent
sys.path.append(str(project_root))

import ast
import pandas as pd
from tqdm import tqdm
import check_6k_mutations_withTable2 as generate_truth

tqdm.pandas()


def get_truth_cnv_and_ssm(cnv_and_ssm):
    cnv_and_ssm['truth'] = cnv_and_ssm.progress_apply(
        lambda x: generate_truth.cnv_and_ssm(
        gene=x['Gene entities'][0], project=x['Cancer entities']
        ), axis=1
    )
    cnv_and_ssm['truth'] = round(cnv_and_ssm['truth'],2)
    return cnv_and_ssm


def get_truth_heterozygous_cnvs(heterozygous_cnvs):
    heterozygous_cnvs['truth'] = heterozygous_cnvs.progress_apply(
        lambda x: generate_truth.freq_cnv_loss_or_gain(
        gene=x['Gene entities'][0], project=x['Cancer entities'][0],mut="Loss"
        ), axis=1
    )
    heterozygous_cnvs['truth'] = round(heterozygous_cnvs['truth'],2)

    return heterozygous_cnvs


def get_truth_homozygous_cnvs(homozygous_cnvs):
    homozygous_cnvs['truth'] = homozygous_cnvs.progress_apply(
        lambda x: generate_truth.freq_cnv_loss_or_gain(
            gene=x['Gene entities'][0], project=x['Cancer entities'][0],mut="Homozygous Deletion"
        ), axis=1
    )
    homozygous_cnvs['truth'] = round(homozygous_cnvs['truth'],2)

    return homozygous_cnvs


def get_truth_combination_heterozygous_cnvs(combination_heterozygous_cnvs):
    combination_heterozygous_cnvs['truth'] = combination_heterozygous_cnvs.progress_apply(
    lambda x: generate_truth.freq_cnv_loss_or_gain_comb(
        gene1=x['Gene entities'][0], gene2=x['Gene entities'][1], project=x['Cancer entities'][0],mut="Loss"
    ), axis=1
    )
    combination_heterozygous_cnvs['truth'] = round(combination_heterozygous_cnvs['truth'], 2)
    return combination_heterozygous_cnvs


def get_truth_combination_homozygous_cnvs(combination_homozygous_cnvs):
    combination_homozygous_cnvs['truth'] = combination_homozygous_cnvs.progress_apply(
    lambda x: generate_truth.freq_cnv_loss_or_gain_comb(
        gene1=x['Gene entities'][0], gene2=x['Gene entities'][1], project=x['Cancer entities'][0],mut="Homozygous Deletion"
    ), axis=1
    )
    combination_homozygous_cnvs['truth'] = round(combination_homozygous_cnvs['truth'], 2)
    return combination_homozygous_cnvs


def get_truth_gains(gains):
    gains['truth'] = gains.progress_apply(
        lambda x: generate_truth.freq_cnv_loss_or_gain(
        gene=x['Gene entities'][0], project=x['Cancer entities'][0],mut="Gain"
    ), axis=1
    )
    gains['truth'] = round(gains['truth'], 2)
    return gains


def get_truth_amplifications(amplifications):
    amplifications['truth'] = amplifications.progress_apply(
        lambda x: generate_truth.freq_cnv_loss_or_gain(
        gene=x['Gene entities'][0], project=x['Cancer entities'][0],mut="Amplification"
    ), axis=1
    )
    amplifications['truth'] = round(amplifications['truth'], 2)
    return amplifications


def get_truth_ssm_freq(ssm_freq):
    ssm_freq['truth'] = ssm_freq.progress_apply(
        lambda x: generate_truth.ssm_frequency(
        gene=x['Gene entities'][0], 
        project=x['Cancer entities'][0],
        mut=x['Mutation entities'][0]
    ), axis=1
    )
    ssm_freq['truth'] = round(ssm_freq['truth'], 2)
    return ssm_freq


def get_truth_msi_freq(msi_freq):
    msi_freq['truth'] = msi_freq.progress_apply(
        lambda x: generate_truth.msi_frequency(
        project=x['Cancer entities'],
    ), axis=1
    )
    msi_freq['truth'] = round(msi_freq['truth'], 2)
    return msi_freq



def main():
    gdc_qag_results = pd.read_csv('csvs/gdc_qag_results.plot.csv')
    gdc_qag_results['Gene entities'] = gdc_qag_results['Gene entities'].apply(
        ast.literal_eval)
    gdc_qag_results['Mutation entities'] = gdc_qag_results['Mutation entities'].apply(
        ast.literal_eval)
    gdc_qag_results['Cancer entities'] = gdc_qag_results['Cancer entities'].apply(
        ast.literal_eval)
    # top_cases_counts_by_gene questions are cnv_and_ssm type
    # merge into one for plotting questions of this type
    gdc_qag_results.loc[gdc_qag_results['Intent'] == 'top_cases_counts_by_gene', 'intent'] = 'cnv_and_ssm'
    
    # cnv_and_ssm intent
    cnv_and_ssm = gdc_qag_results[gdc_qag_results['Question'].str.contains('simple somatic mutations or copy number variants')]
    print('obtaining truth results for cnv_and_ssm...')
    print('cnv_and_ssm shape {}'.format(cnv_and_ssm.shape))
    
    # test
    # cnv_and_ssm = cnv_and_ssm.head(n=2)
    cnv_and_ssm = get_truth_cnv_and_ssm(cnv_and_ssm)
    print('completed cnv_and_ssm')

    # heterozygous cnvs
    cnvs = gdc_qag_results[gdc_qag_results['Intent'] == 'freq_cnv_loss_or_gain']
    heterozygous_cnvs = cnvs[cnvs['Question'].str.contains('heterozygous')]
    print('obtaining truth results for heterozygous cnvs...')
    print('heterozygous cnvs shape {}'.format(heterozygous_cnvs.shape))
    
    # test
    # heterozygous_cnvs = heterozygous_cnvs.head(n=2)
    heterozygous_cnvs = get_truth_heterozygous_cnvs(heterozygous_cnvs)
    heterozygous_cnvs.to_csv('csvs/heterozygous_cnvs.truth.csv')
    print('completed heterozygous cnvs')

    # homozygous cnvs
    print('obtaining truth results for homozygous cnvs...')
    homozygous_cnvs = cnvs[cnvs['Question'].str.contains('homozygous')]
    print('homozygous cnv shape {}'.format(homozygous_cnvs.shape))
    homozygous_cnvs = get_truth_homozygous_cnvs(homozygous_cnvs)
    print('completed homozygous cnvs')

    # gains
    print('obtaining truth results for gains...')
    gains = cnvs[cnvs['Question'].str.contains('gain')]
    print('gains shape: {}'.format(gains.shape))
    
    # test
    # gains = gains.head(n=2)
    gains = get_truth_gains(gains)
    print('completed gains')

    # amplifications
    print('obtaining truth results for amplifications...')
    amplifications = cnvs[cnvs['Question'].str.contains('amplification')]
    print('amplifications shape: {}'.format(amplifications.shape))
    
    # test
    # amplifications = amplifications.head(n=2)
    amplifications = get_truth_amplifications(amplifications)
    print('completed amplifications')
    
    # ssm freq
    print('obtaining truth results for ssm freq...')
    ssm_freq = gdc_qag_results[gdc_qag_results['Intent'] == 'ssm_frequency']
    print('ssm freq shape {}'.format(ssm_freq.shape))
    
    # test
    # ssm_freq = ssm_freq.head(n=2)
    ssm_freq = get_truth_ssm_freq(ssm_freq)
    print('completed ssm freq')

    # msi freq
    print('obtaining truth results for msi freq...')
    msi_freq = gdc_qag_results[gdc_qag_results['Intent'] == 'msi_h_frequency']
    print('msi freq shape {}'.format(msi_freq.shape))
    # test
    # msi_freq = msi_freq.head(n=2)
    msi_freq = get_truth_msi_freq(msi_freq)
    print('completed msi freq')

    # combination heterozygous cnvs
    combination_cnvs = gdc_qag_results[gdc_qag_results['Intent'] == 'freq_cnv_loss_or_gain_comb']
    print('obtaining truth results for combination heterozygous cnvs...')
    combination_heterozygous_cnvs = combination_cnvs[combination_cnvs['Question'].str.contains('co-occurence frequency of somatic heterozygous deletions')]
    print('combination heterozygous cnvs shape {}'.format(combination_heterozygous_cnvs.shape))
    
    # test
    # combination_heterozygous_cnvs = combination_heterozygous_cnvs.head(n=2)
    combination_heterozygous_cnvs = get_truth_combination_heterozygous_cnvs(combination_heterozygous_cnvs)
    print('completed combination heterozygous cnvs')

    # combination homozygous cnvs
    print('obtaining truth results for combination homozygous cnvs...')
    combination_homozygous_cnvs = combination_cnvs[combination_cnvs['Question'].str.contains('co-occurence frequency of somatic homozygous deletions')]
    print('combination homozygous cnvs shape {}'.format(combination_homozygous_cnvs.shape))
    combination_homozygous_cnvs = get_truth_combination_homozygous_cnvs(combination_homozygous_cnvs)
    print('completed combination homozygous cnvs')

    print('finished, concatenating results')
    merged = pd.concat([
        cnv_and_ssm,
        heterozygous_cnvs,
        homozygous_cnvs,
        gains,
        amplifications,
        combination_heterozygous_cnvs,
        combination_homozygous_cnvs,
        ssm_freq,
        msi_freq
    ])
    required_columns = ['Question', 'truth']
    merged.to_csv('csvs/truth.csv', columns=required_columns)
    print('finished writing results')



if __name__ == '__main__':
    main()


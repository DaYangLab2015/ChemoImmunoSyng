#!/usr/bin/env python
# coding: utf-8
# Corr. Author: Yue Wang (yuw90@pitt.edu)

import matplotlib
import sys
import os

import pandas as pd
import numpy as np
import seaborn as sns
import gseapy as gp

from scipy import stats
from pandas import DataFrame as df
from matplotlib import pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# # Utilities

def merge_prerank(result_sets):
    '''
    Merge the prerank results into matrix
    Input:
        result_sets: dict, {sampels: prerank results}
    Return:
        matrix_sets: dict, {NES, ES, p-val}
    '''
    # extract signature
    sig_sets = list(result_sets[list(result_sets.keys())[0]]['Term'])
    
    # initialize
    matrix_sets = {'NES': df(columns=sig_sets),
                   'ES': df(columns=sig_sets),
                   'FDR q-val': df(columns=sig_sets)}
    
    for p in result_sets.keys():
        tmp_p = result_sets[p].set_index('Term')
        for s in tmp_p.index:
            for k in matrix_sets:
                matrix_sets[k].at[p, s] = tmp_p.loc[s, k]
    
    return matrix_sets


def calculate_RS_score(X, sig_R, sig_S, method='average'):
    '''
    Calculate RS score for given R signature and S signature.
    !!!Notice: RS score is equivalent to the calculation of shift ability when method is "gsea"!!!
    Input:
        X: pandas dataframe, index=samples, columns=features
        sig_R, sig_S: list of genes in each signature
        method: str, method to calculate the RS score, ['average', 'gsea']
    Return:
        score_: pandas dataframe, index=samples, columns=['R_score', 'S_score', 'RS_score']
    '''
    if method == 'average':
        R_ = X.T[X.columns.isin(sig_R)].T.mean(axis=1)
        S_ = X.T[X.columns.isin(sig_S)].T.mean(axis=1)
        RS_ = R_ - S_
        
    elif method == 'gsea':
        gene_set = {'R_signature': list(sig_R),
                    'S_signature': list(sig_S)}
        tmp_ = {}
        for p in X.index:
            tmp_[p] = gp.prerank(X.T[p],
                                 gene_sets=gene_set,
                                 threads=16,
                                 permutation_num=200,
                                 outdir=None,
                                 no_plot=True,
                                 seed=0,
                                 min_size=0,
                                 max_size=500).res2d
        merged_ = merge_prerank(tmp_)
        R_ = merged_['NES']['R_signature']
        S_ = merged_['NES']['S_signature']
        RS_ = R_ - S_
    else:
        raise ValueError("Unexpected method: " + method + ". Must be 'average' or 'gsea'.")
    
    # merge results
    score_ = df(index=X.index, columns=['R_score', 'S_score', 'RS_score'])
    score_['R_score'] = R_
    score_['S_score'] = S_
    score_['RS_score'] = RS_
    
    return score_


# # Parameter sets

# for input and output
path_to_X = '/path/to/post/treatment/transcriptome/file'
path_to_annotation = '/path/to/gene/annotation/file'
path_to_result = 'path/to/result/output/folder'
gsea_home = 'path/to/core/signature/set/folder'
output_name = 'output'


# # 0. Read X

X = pd.read_csv(path_to_X, header=0, index_col=0, sep=',')

bing_landmark = pd.read_csv(path_to_annotation,
                            header=0, index_col=0, sep=',',
                            dtype={'Official NCBI gene symbol': 'str'},
                            converters={'Official NCBI gene symbol': None})

X.index = X.index.astype(int)
X = X[X.index.isin(bing_landmark.index)].rename(index=bing_landmark['Official NCBI gene symbol']).T

# Create output dir
if output_name not in os.listdir(path_to_result):
    os.mkdir(path_to_result + output_name)

# Read gene set
gene_sets = {}
with open(gsea_home + '/gene_sets/core_RS_signature.gmt', 'r') as f:
    for lines in f:
        lines = lines.rstrip().split('\t')
        gene_sets[lines[0]] = lines[2:]


# # 1. Calculate RS score
# Average score
RS_average = calculate_RS_score(X=X,
                                sig_R=gene_sets['R_signature'],
                                sig_S=gene_sets['S_signature'],
                                method='average')


# Shift ability score
RS_gsea = calculate_RS_score(X=X,
                             sig_R=gene_sets['R_signature'],
                             sig_S=gene_sets['S_signature'],
                             method='gsea')


# # 2. Save RS score
# save file
RS_average.to_csv(path_to_result + output_name + '/RS_score_by_average.csv', sep=',')
RS_gsea.to_csv(path_to_result + output_name + '/RS_score_by_gsea.csv', sep=',')
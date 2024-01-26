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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from scipy import stats
from pandas import DataFrame as df
from matplotlib import pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# # Utilities
def split_validation(X, y_binary, **kwargs):
    '''
    Split the training set and validation set.
    Input:
        X: pandas dataframe, index=samples, columns=features
        y: pandas dataframe, index=samples, columns=label
        kwargs: kwargs pass to sklearn.model_selection.tran_test_split
    Return:
        X_train, y_train: training set
        X_validation, y_validation: validation set
    '''
    if 'stratify' not in kwargs:
        stratify = y_binary
    if 'random_state' not in kwargs:
        random_state = 0
    if 'test_size' not in kwargs:
        test_size = 0.25
    # split the training and validation
    X_train, X_validation, y_train, y_validation = train_test_split(X, y_binary,
                                                                    test_size=kwargs['kwargs']['test_size'],
                                                                    random_state=kwargs['kwargs']['random_state'],
                                                                    stratify=kwargs['kwargs']['stratify'])

    return X_train, X_validation, y_train, y_validation


def kfold_cv(X, y, k_fold, **kwargs):
    '''
    Split the X into k fold cv set.
    Input:
        X: pandas dataframe, index=samples, columns=features
        y: pandas dataframe, index=samples, columns=label(str)
        kwargs: kwargs pass to sklearn.model_selection.StratifiedKFold
    Return:
        skf_set: dict, {'i_fold': {'X_train', 'y_train', 'X_test', 'y_test'}, ...}
    '''
    if 'random_state' not in kwargs:
        random_state = None
    if 'shuffle' not in kwargs:
        shuffle = False

    skf = StratifiedKFold(n_splits=k_fold, random_state=random_state)
    skf_set = {}
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        skf_set['fold_' + str(i)] = {'X_train': X.iloc[train_index, :],
                                     'y_train': y.iloc[train_index, :],
                                     'X_test': X.iloc[test_index, :],
                                     'y_test': y.iloc[test_index, :]}
    return skf_set


def bootstrapping_DEG(X, y, n_bootstrap, **kwargs):
    '''
    DEG analysis conjugated with bootstrapping.
    Input:
        X: pandas dataframe, index=samples, columns=features
        y: pandas dataframe, index=samples, columns=label(str)
        n_bootstrap: number of bootstrapping times
        kwargs: kwargs pass to DEG function
    Return:
        selection_matrix: pandas dataframe, index=features, columns=['R_select', 'S_select']
    '''
    # initialize
    selection_matrix = df(index=X.columns, columns=['R_select', 'S_select']).fillna(0.)

    for i in range(n_bootstrap):
        X_, y_ = resample(X, y, replace=True, stratify=y, random_state=None)
        selection_matrix += DEG(X=X_, y=y_, kwargs=kwargs['kwargs'])

    selection_matrix = selection_matrix / n_bootstrap

    return selection_matrix


def DEG(X, y, **kwargs):
    '''
    Perform DEG analysis with given X and y
    Input:
        X: pandas dataframe, index=samples, columns=features
        y: pandas dataframe, index=samples, columns=label(str)
        kwargs: method=['ttest', 'ranksum'], fc_cutoff, p_cutoff
    Return:
        selection_vector: pandas dataframe, index=genes, columns=['R_select', 'S_select'], binary
    '''
    # group patients
    y = y.astype(str)
    tmp_R = y[y['response'] == '1'].index
    tmp_S = y[y['response'] == '0'].index

    # initialize
    deg_result = df(index=X.columns, columns=['p_value', 'log2FC'])
    # run DEG analysis
    if kwargs['kwargs']['method'] == 'ttest':
        deg_result['p_value'] = X.apply(lambda x: stats.ttest_ind(x.loc[tmp_R], x.loc[tmp_S], equal_var=False)[1])
    elif kwargs['kwargs']['method'] == 'ranksum':
        deg_result['p_value'] = X.apply(lambda x: stats.ranksums(x.loc[tmp_R], x.loc[tmp_S])[1])
    else:
        raise ValueError('Expected method to be ttest or ranksum.')

    # average foldchange calculation: default is S - R
    deg_result['log2FC'] = X.apply(lambda x: x.loc[tmp_R].mean() - x.loc[tmp_S].mean())

    # assign significance
    selection_vector = df(index=deg_result.index, columns=['R_select', 'S_select'])
    tmp_sig = deg_result[deg_result['p_value'] <= kwargs['kwargs']['p_cutoff']]
    tmp_sig = tmp_sig[abs(tmp_sig['log2FC']) > kwargs['kwargs']['fc_cutoff']]
    selection_vector.at[tmp_sig[tmp_sig['log2FC'] > 0.].index, 'R_select'] = 1
    selection_vector.at[tmp_sig[tmp_sig['log2FC'] < 0.].index, 'S_select'] = 1
    selection_vector = selection_vector.fillna(0)

    return selection_vector


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


def calculate_RS_score(X, y, sig_R, sig_S, method='average'):
    '''
    Calculate RS score for given R signature and S signature.
    Notice: RS score is equivalent to the calculation of shift ability when method is "gsea"
    Input:
        X: pandas dataframe, index=samples, columns=features
        y: pandas dataframe, index=samples, columns=label(str)
        sig_R, sig_S: list of genes in each signature
        method: str, method to calculate the RS score, ['average', 'gsea']
    Return:
        score_: pandas dataframe, index=samples, columns=['R_score', 'S_score', 'RS_score']
        auc_: dict, {'R_score', 'S_score', 'RS_score'}
    '''
    if method == 'average':
        R_ = X.T[X.columns.isin(sig_R)].T.mean(axis=1)
        S_ = X.T[X.columns.isin(sig_S)].T.mean(axis=1)
        RS_ = R_ - S_ # equivalent to shift ability when method="gsea"

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
                                 max_size=5000).res2d
        merged_ = merge_prerank(tmp_)
        R_ = merged_['NES']['R_signature']
        S_ = merged_['NES']['S_signature']
        RS_ = R_ - S_
    else:
        raise ValueError("Unexpected method: " + method + ". Must be 'average' or 'gsea'.")

    # merge results
    score_ = df(index=X.index, columns=['R_score', 'S_score', 'RS_score'])
    score_['R_score'] = R_
    score_['S_score'] = -S_
    score_['RS_score'] = RS_

    y_sorted = y.loc[score_.index]

    # calculate AUC
    auc_ = {}
    auc_['R_score'] = AUC(y=y_sorted, score=score_['R_score'])
    auc_['S_score'] = AUC(y=y_sorted, score=score_['S_score'])
    auc_['RS_score'] = AUC(y=y_sorted, score=score_['RS_score'])

    return score_, auc_


def AUC(y, score):
    '''
    Calculate AUC for the given score
    '''
    auc_ = roc_auc_score(y_true=y, y_score=score)

    return auc_


def select_sig_by_cv(sigma_set, cv_set, top_start, top_end, top_step, method='average'):
    '''
    Selection score cutoff by cross-validation: average method
    Input:
        sigma_set: dict, sets of signature selection matrix derived from bootstrappingDEG
        cv_set: dict, sets of expression matrix derived from kfold_cv
        top_start: float, cutoff start
        top_end: float, cutoff end
        top_step: float, spacing between start and end
        method: str, method to pass to calculate_RS_score
    Return:
        R_train, R_test, S_train, S_test, RS_train, RS_test: AUC of signature performance under each condition
    '''
    # initialize the cutoff selection
    selection_cutoff = list(np.arange(top_start, top_end, top_step))

    # initialize the output matrix
    R_train = df(index=selection_cutoff, columns=sigma_kfold.keys())
    R_test = df(index=selection_cutoff, columns=sigma_kfold.keys())
    S_train = df(index=selection_cutoff, columns=sigma_kfold.keys())
    S_test = df(index=selection_cutoff, columns=sigma_kfold.keys())
    RS_train = df(index=selection_cutoff, columns=sigma_kfold.keys())
    RS_test = df(index=selection_cutoff, columns=sigma_kfold.keys())

    # For each cutoff, construct tmp_Rsig and tmp_Ssig in each fold
    for e in selection_cutoff:
        for k in sigma_set.keys():
            tmp_sigma = sigma_set[k]

            # tmp_R cutoff
            tmp_R_quant = tmp_sigma['R_select'].quantile(1 - e)
            tmp_R = tmp_sigma[tmp_sigma['R_select'] > tmp_R_quant].index

            # tmp_S cutoff
            tmp_S_quant = tmp_sigma['S_select'].quantile(1 - e)
            tmp_S = tmp_sigma[tmp_sigma['S_select'] > tmp_S_quant].index

            if tmp_S.shape[0] != 0 and tmp_R.shape[0] != 0:
                # training auc
                _, train_auc_ = calculate_RS_score(X=cv_sets[k]['X_train'],
                                                   y=cv_sets[k]['y_train'],
                                                   sig_R=tmp_R,
                                                   sig_S=tmp_S,
                                                   method=method)
                # test auc
                _, test_auc_ = calculate_RS_score(X=cv_sets[k]['X_test'],
                                                  y=cv_sets[k]['y_test'],
                                                  sig_R=tmp_R,
                                                  sig_S=tmp_S,
                                                  method=method)
                # write to output matrix
                R_train.at[e, k] = train_auc_['R_score']
                R_test.at[e, k] = test_auc_['R_score']
                S_train.at[e, k] = train_auc_['S_score']
                S_test.at[e, k] = test_auc_['S_score']
                RS_train.at[e, k] = train_auc_['RS_score']
                RS_test.at[e, k] = test_auc_['RS_score']

    return R_train, R_test, S_train, S_test, RS_train, RS_test


def val_by_select(X_val, y_val, sigma_set, top_start, top_end, top_step, method='average'):
    '''
    Validation performance at each selection score cutoff: average method
    Input:
        X_val: pandas dataframe, index=samples, columns=features
        y_val: pandas dataframe, index=samples, columns=label(str)
        sigma_set: dict, sets of signature selection matrix derived from bootstrappingDEG
        top_start: float, cutoff start
        top_end: float, cutoff end
        top_step: float, spacing between start and end
        method: str, method to pass to calculate_RS_score
    Return:
        R_val, S_val, RS_val: AUC of signature performance under each condition
    '''
    # initialize the cutoff selection
    selection_cutoff = list(np.arange(top_start, top_end, top_step))

    # initialize the output matrix
    R_val = df(index=selection_cutoff, columns=sigma_kfold.keys())
    S_val = df(index=selection_cutoff, columns=sigma_kfold.keys())
    RS_val = df(index=selection_cutoff, columns=sigma_kfold.keys())

    # For each cutoff, construct tmp_Rsig and tmp_Ssig in each fold
    for e in selection_cutoff:
        for k in sigma_set.keys():
            tmp_sigma = sigma_set[k]

            # tmp_R cutoff
            tmp_R_quant = tmp_sigma['R_select'].quantile(1 - e)
            tmp_R = tmp_sigma[tmp_sigma['R_select'] > tmp_R_quant].index

            # tmp_S cutoff
            tmp_S_quant = tmp_sigma['S_select'].quantile(1 - e)
            tmp_S = tmp_sigma[tmp_sigma['S_select'] > tmp_S_quant].index

            if tmp_S.shape[0] != 0 and tmp_R.shape[0] != 0:
                # training auc
                _, val_auc_ = calculate_RS_score(X=X_val,
                                                 y=y_val['response'].astype(int),
                                                 sig_R=tmp_R,
                                                 sig_S=tmp_S,
                                                 method=method)
                R_val.at[e, k] = val_auc_['R_score']
                S_val.at[e, k] = val_auc_['S_score']
                RS_val.at[e, k] = val_auc_['RS_score']

    return R_val, S_val, RS_val


def visual_cv(score_train, score_test, title=None, save_path=None, **kwargs):
    '''
    Visualize the cross-validation performance.
    Input:
        score_train, score_test: pandas dataframe, auc matrix of training set and test set, index=cutoff, columns=fold
        title: str, title of the plot
        save_path: str, path/to/save/the/plot
        kwargs: kwargs pass to sns.lineplot()
    Return:
        None
    '''
    # assign cutoff to a new column
    tmp_train = score_train.copy()
    tmp_test = score_test.copy()
    tmp_train['cutoff'] = score_train.index.astype(float)
    tmp_test['cutoff'] = score_test.index.astype(float)

    # wide to long
    score_train_long = pd.melt(tmp_train, id_vars='cutoff')
    score_test_long = pd.melt(tmp_test, id_vars='cutoff')

    # visual by seaborn lineplot
    if 'figsize' not in kwargs:
        figsize = (4, 3)

    plt.figure(figsize=figsize)
    sns.lineplot(data=score_train_long,
                 x='cutoff', y='value',
                 ci=None,
                 label='training',
                 color='crimson')

    sns.lineplot(data=score_test_long,
                 x='cutoff', y='value',
                 ci=None,
                 label='test',
                 color='royalblue')
    plt.xlabel('selection cutoff')
    plt.ylabel('auc')

    # finalize
    if title != None:
        plt.title(title)
        plt.tight_layout()
    if save_path != None:
        fig = plt.gcf()
        fig.savefig(save_path + '.png', dpi=300, transparent=True)
        fig.savefig(save_path + '.pdf', transparent=True)

    plt.show()
    return


def select_RS_by_cv(sigma_set, cv_set, top_start, top_end, top_step, method='average'):
    '''
    Selection R cutoff and S cutoff by cross-validation: average method
    Different from select_sig_by_cv(...), in which the cutoff for R and S is the same.
    Input:
        sigma_set: dict, sets of signature selection matrix derived from bootstrappingDEG
        cv_set: dict, sets of expression matrix derived from kfold_cv
        top_start: float, cutoff start
        top_end: float, cutoff end
        top_step: float, spacing between start and end
        method: str, method to pass to calculate_RS_score
    Return:
        RS_train, RS_test: average AUC of RS score performance under each condition
    '''
    # initialize the cutoff selection
    R_cutoff = list(np.arange(top_start, top_end, top_step))
    S_cutoff = list(np.arange(top_start, top_end, top_step))

    # initialize the output matrix
    RS_train = df(index=R_cutoff, columns=S_cutoff)
    RS_test = df(index=R_cutoff, columns=S_cutoff)

    # For each R and S cutoff, construct tmp_Rsig and tmp_Ssig in each fold
    for r in R_cutoff:
        for s in S_cutoff:
            tmp_RS_train = []
            tmp_RS_test = []

            # calculate auc in each fold
            for k in sigma_set.keys():
                tmp_sigma = sigma_set[k]

                # tmp_R cutoff
                tmp_R_quant = tmp_sigma['R_select'].quantile(1 - r)
                tmp_R = tmp_sigma[tmp_sigma['R_select'] > tmp_R_quant].index

                # tmp_S cutoff
                tmp_S_quant = tmp_sigma['S_select'].quantile(1 - s)
                tmp_S = tmp_sigma[tmp_sigma['S_select'] > tmp_S_quant].index

                if tmp_S.shape[0] != 0 and tmp_R.shape[0] != 0:
                    # training auc
                    _, train_auc_ = calculate_RS_score(X=cv_sets[k]['X_train'],
                                                       y=cv_sets[k]['y_train'],
                                                       sig_R=tmp_R,
                                                       sig_S=tmp_S,
                                                       method=method)
                    # test auc
                    _, test_auc_ = calculate_RS_score(X=cv_sets[k]['X_test'],
                                                      y=cv_sets[k]['y_test'],
                                                      sig_R=tmp_R,
                                                      sig_S=tmp_S,
                                                      method=method)
                    tmp_RS_train.append(train_auc_['RS_score'])
                    tmp_RS_test.append(test_auc_['RS_score'])
                # write to output matrix
                RS_train.at[r, s] = sum(tmp_RS_train) / len(tmp_RS_train)
                RS_test.at[r, s] = sum(tmp_RS_test) / len(tmp_RS_test)

    return RS_train, RS_test


def core_sig_idf(sigma_set, R_cutoff, S_cutoff, X_val, y_val, validation_type, do_pathway=False, do_venn=True, save_path=None):
    '''
    Generate R sig and S sig based on given cutoff.
    Input:
        sigma_set: dict, sets of signature selection matrix derived from bootstrappingDEG
        R_cutoff, S_cutoff: float, top x percentage of gene selection
        X_val, y_val: validation set
        validation_type: str, label of validation set.
        do_pathway: bool, default False. If true, do enrichment pathway analysis using GO terms
        do_venn: bool, default True. If true, draw the venn for signatures from each fold
        save_path: path to result file
    Return:
        R_core, S_core: list of signature genes under the corresponding cutoff
        validation_auc: validation performance of core signature
    '''
    # get signature
    R_set = []
    S_set = []
    for k in sigma_set.keys():
        tmp_sigma = sigma_set[k]

        # tmp_R cutoff
        tmp_R_quant = tmp_sigma['R_select'].quantile(1 - R_cutoff)
        tmp_R = tmp_sigma[tmp_sigma['R_select'] > tmp_R_quant].index

        # tmp_S cutoff
        tmp_S_quant = tmp_sigma['S_select'].quantile(1 - S_cutoff)
        tmp_S = tmp_sigma[tmp_sigma['S_select'] > tmp_S_quant].index

        R_set.append(set(tmp_R))
        S_set.append(set(tmp_S))

    # venn: R
    venn3(R_set)
    plt.title('R sig genes in each fold')
    fig_R = plt.gcf()
    plt.show()

    # venn: S
    venn3(S_set)
    plt.title('S sig genes in each fold')
    fig_S = plt.gcf()
    plt.show()
    
    # core R
    R_core = list(R_set[0] & R_set[1] & R_set[2])
    S_core = list(S_set[0] & S_set[1] & S_set[2])
    

    # score result initiation
    cv_score = df(index=cv_sets.keys(),
                  columns=['RS_train', 'RS_test', 'R_train',
                           'S_train', 'R_test', 'S_test'])

    for k in cv_sets.keys():
        train_X_ = cv_sets[k]['X_train']
        train_y_ = cv_sets[k]['y_train']
        test_X_ = cv_sets[k]['X_test']
        test_y_ = cv_sets[k]['y_test']
        
    
        _, train_auc_ = calculate_RS_score(X=train_X_,
                                           y=train_y_['response'].astype(int),
                                           sig_R=R_core,
                                           sig_S=S_core,
                                           method=method)
        _, test_auc_ = calculate_RS_score(X=test_X_,
                                          y=test_y_['response'].astype(int),
                                          sig_R=R_core,
                                          sig_S=S_core,
                                          method=method)
        cv_score.at[k, 'RS_train'] = train_auc_['RS_score']
        cv_score.at[k, 'R_train'] = train_auc_['R_score']
        cv_score.at[k, 'S_train'] = train_auc_['S_score']
        cv_score.at[k, 'RS_test'] = test_auc_['RS_score']
        cv_score.at[k, 'R_test'] = test_auc_['R_score']
        cv_score.at[k, 'S_test'] = test_auc_['S_score']
    
    return R_core, S_core, cv_score


"""
Parameter sets
"""
# for input and output
path_to_X = 'path/to/X.csv'
path_to_y = 'path/to/y.csv'
path_to_result = 'path/to/result/'
output_name = 'output_name'

# for spliting the training and validation set
split_random_state = 0
split_test_size = .2

# for cross-validation and bootstrapping
n_bootstrap = 100 # [10-1000]
k_fold = 3 # [3-5]
top_step = 0.01 # step for selecting top genes
top_start = 0.01 # top 1% genes
top_end = 0.25 # top 10% genes

# for DEG criteria
kwargs_DEG = {'method': 'ttest',
              'fc_cutoff': 0., # set to 0 if only want to use p value as cutoff
              'p_cutoff': 0.05}


"""
0. Preprocessing of X and y
"""

X = pd.read_csv(path_to_X, index_col=0, header=0, sep=',') # X is preprocessed (normalization, low variance filtering)
print(X.shape)
print(X.head())

y = pd.read_csv(path_to_y, index_col=0, header=0, sep=',') # y is preprocessed (RECIST evaluation)
y = y[y.index.isin(X.columns)].groupby(level=0).first()
print(y.shape)
print(y.head())
print(y['Resp_NoResp'].value_counts())

# transform the response into binary value {'No_Response': 1, 'Response': 0}
y_binary = df(index=y.index, columns=['response'])
y_binary.at[y[y['Resp_NoResp'] == 'No_Response'].index, 'response'] = 1
y_binary.at[y[y['Resp_NoResp'] == 'Response'].index, 'response'] = 0

# SD grouped with PRCR (optional)
y_binary.at[y[y['BOR'] == 'SD'].index, 'response'] = 0
print(y_binary.head())

y_binary = df(y_binary.loc[X.columns, :])


"""
1. Split the validation set
"""

X_train, X_val, y_train, y_val = split_validation(X.T, y_binary, kwargs={'stratify': y_binary,
                                                                         'random_state': split_random_state,
                                                                         'test_size': split_test_size})

# Save to _tmp under the path/to/results
if output_name not in os.listdir(path_to_result):
    os.mkdir(path_to_result + output_name)

X_train.to_csv(path_to_result + output_name + '/X_train.csv', sep=',')
X_val.to_csv(path_to_result + output_name + '/X_validation.csv', sep=',')
y_train.to_csv(path_to_result + output_name + '/y_train.csv', sep=',')
y_val.to_csv(path_to_result + output_name + '/y_validation.csv', sep=',')


# __Note__: The training will be only performed on X_train. The X_val will be only used for validation.

"""
2. Signature construction with k-fold CV + resampling
"""

# ## 2.1 Stratified k-fold CV split

cv_sets = kfold_cv(X=X_train, y=y_train.astype(str), k_fold=k_fold)

# Save to _tmp under the path/to/results
if 'cv' not in os.listdir(path_to_result + output_name + '/'):
    os.mkdir(path_to_result + output_name + '/cv')

for k in cv_sets:
    if k not in os.listdir(path_to_result + output_name + '/cv/'):
        os.mkdir(path_to_result + output_name + '/cv/' + k)
        cv_sets[k]['X_train'].to_csv(path_to_result + output_name + '/cv/' + k + '/X_train.csv', sep=',')
        cv_sets[k]['y_train'].to_csv(path_to_result + output_name + '/cv/' + k + '/y_train.csv', sep=',')
        cv_sets[k]['X_test'].to_csv(path_to_result + output_name + '/cv/' + k + '/X_test.csv', sep=',')
        cv_sets[k]['y_test'].to_csv(path_to_result + output_name + '/cv/' + k + '/y_test.csv', sep=',')

# ## 2.2 DEG with resampling
# ---------------------------------demo run-----------------------------------------
# ### 2.2.1 Run test

test = bootstrapping_DEG(X=cv_sets['fold_1']['X_train'],
                         y=cv_sets['fold_1']['y_train'],
                         n_bootstrap=10,
                         kwargs=kwargs_DEG)

test_Rsig = test[test['R_select'] >= test['R_select'].quantile(0.9)].index
test_Ssig = test[test['S_select'] >= test['S_select'].quantile(0.9)].index
print(test_Rsig.shape, test_Ssig.shape)

score_, auc_ = calculate_RS_score(X=cv_sets['fold_1']['X_test'],
                                  y=cv_sets['fold_1']['y_test'],
                                  sig_R=test_Rsig,
                                  sig_S=test_Ssig,
                                  method='average')
print(auc_)
sns.scatterplot(x=score_['RS_score'], y=cv_sets['fold_1']['y_test']['response'].astype(int))
plt.show()

score_, auc_ = calculate_RS_score(X=cv_sets['fold_1']['X_train'],
                                  y=cv_sets['fold_1']['y_train'],
                                  sig_R=test_Rsig,
                                  sig_S=test_Ssig,
                                  method='average')
print(auc_)
sns.scatterplot(x=score_['RS_score'], y=cv_sets['fold_1']['y_train']['response'].astype(int))
plt.show()

score_, auc_ = calculate_RS_score(X=X_val,
                                  y=y_val['response'].astype(int),
                                  sig_R=test_Rsig,
                                  sig_S=test_Ssig,
                                  method='average')
print(auc_)
sns.scatterplot(x=score_['RS_score'], y=y_val['response'].astype(int))
plt.show()


# ## 2.2 Run on all the k-fold set
# -------------------------------Training run--------------------------------------------
'''
1. Generate R_i and S_i sig selection matrix through bootstrapping
'''
# this dictionary is used to save signature selection matrix for each fold in CV
sigma_kfold = {}

for k in cv_sets.keys():
    print('Running bootstrappingDEG on ' + k + '...')
    print('Bootstrapping times set to ' + str(n_bootstrap) + '...')
    sigma_kfold[k] = bootstrapping_DEG(X=cv_sets[k]['X_train'],
                                       y=cv_sets[k]['y_train'],
                                       n_bootstrap=n_bootstrap,
                                       kwargs=kwargs_DEG)

# save to tmp
if 'bootstrappingDEG' not in os.listdir(path_to_result + output_name + '/'):
    os.mkdir(path_to_result + output_name + '/bootstrappingDEG/')
for k in sigma_kfold.keys():
    sigma_kfold[k].to_csv(path_to_result + output_name + '/bootstrappingDEG/selection_score_' + k + '.csv', sep=',')


for k in sigma_kfold.keys():
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x='R_select', y='S_select', data=sigma_kfold[k], s=5, color='crimson')
    plt.title(k)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(path_to_result + output_name + '/bootstrappingDEG/' + k + '.png', transparent=True, dpi=300)
    fig.savefig(path_to_result + output_name + '/bootstrappingDEG/' + k + '.pdf', transparent=True)


# overlapped genes between folds
op_gene_fold = df(index=list(np.arange(0.0, 1, 0.01)), columns=['R_genes', 'S_genes'])
kf1 = sigma_kfold['fold_0']
kf2 = sigma_kfold['fold_1']
kf3 = sigma_kfold['fold_2']

for i in op_gene_fold.index:
    op_gene_fold.at[i, 'R_genes'] = len(list(set(kf1[kf1['R_select'] >= i].index) & set(kf2[kf2['R_select'] >= i].index) & set(kf3[kf3['R_select'] >= i].index)))
    op_gene_fold.at[i, 'S_genes'] = len(list(set(kf1[kf1['S_select'] >= i].index) & set(kf2[kf2['S_select'] >= i].index) & set(kf3[kf3['S_select'] >= i].index)))


plt.figure(figsize=(4, 4))
sns.scatterplot(x=op_gene_fold.index, y='R_genes', data=op_gene_fold, s=5, color='crimson')
sns.scatterplot(x=op_gene_fold.index, y='S_genes', data=op_gene_fold, s=5, color='royalblue')
plt.ylabel('gene number')
plt.xlabel('selection score cutoff')
plt.tight_layout()
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/bootstrappingDEG/gene_number_by_cutoff.png', transparent=True, dpi=300)
fig.savefig(path_to_result + output_name + '/bootstrappingDEG/gene_number_by_cutoff.pdf', transparent=True)


'''
2. Selection score cutoff by cross-validation: average method
'''
# average method
R_train_ave, R_test_ave, S_train_ave, S_test_ave, RS_train_ave, RS_test_ave = select_sig_by_cv(sigma_set=sigma_kfold,
                                                                                               cv_set=cv_sets,
                                                                                               top_start=top_start,
                                                                                               top_end=top_end,
                                                                                               top_step=top_step,
                                                                                               method='average')

if 'cv_performance' not in os.listdir(path_to_result + output_name + '/'):
    os.mkdir(path_to_result + output_name + '/cv_performance/')

# RS score
visual_cv(score_train=RS_train_ave,
          score_test=RS_test_ave,
          title='average CV performance of RS score',
          save_path=path_to_result + output_name + '/cv_performance/RS_score_ave')
RS_train_ave.to_csv(path_to_result + output_name + '/cv_performance/RS_train_ave.csv', sep=',')
RS_test_ave.to_csv(path_to_result + output_name + '/cv_performance/RS_test_ave.csv', sep=',')

# R score
visual_cv(score_train=R_train_ave,
          score_test=R_test_ave,
          title='average CV performance of R score',
          save_path=path_to_result + output_name + '/cv_performance/R_score_ave')
R_train_ave.to_csv(path_to_result + output_name + '/cv_performance/R_train_ave.csv', sep=',')
R_test_ave.to_csv(path_to_result + output_name + '/cv_performance/R_test_ave.csv', sep=',')

# S score
visual_cv(score_train=S_train_ave,
          score_test=S_test_ave,
          title='average CV performance of S score',
          save_path=path_to_result + output_name + '/cv_performance/S_score_ave')
S_train_ave.to_csv(path_to_result + output_name + '/cv_performance/S_train_ave.csv', sep=',')
S_test_ave.to_csv(path_to_result + output_name + '/cv_performance/S_test_ave.csv', sep=',')

# Performance in validation set
R_val_ave, S_val_ave, RS_val_ave = val_by_select(X_val=X_val,
                                                 y_val=y_val,
                                                 sigma_set=sigma_kfold,
                                                 top_start=top_start,
                                                 top_end=top_end,
                                                 top_step=top_step,
                                                 method='average')

R_val_ave.to_csv(path_to_result + output_name + '/cv_performance/R_val_ave.csv', sep=',')
S_val_ave.to_csv(path_to_result + output_name + '/cv_performance/S_val_ave.csv', sep=',')
RS_val_ave.to_csv(path_to_result + output_name + '/cv_performance/RS_val_ave.csv', sep=',')

plt.figure(figsize=(3, 2))
sns.kdeplot(R_val_ave.mean(axis=1), label='R_in_validation')
sns.kdeplot(S_val_ave.mean(axis=1), label='S_in_validation')
sns.kdeplot(RS_val_ave.mean(axis=1), label='RS_in_validation')
plt.legend()
plt.tight_layout()
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/cv_performance/validation_performance_ave.pdf', transparent=True)
fig.savefig(path_to_result + output_name + '/cv_performance/validation_performance_ave.png', transparent=True, dpi=300)


'''
3. Select R cutoff and S cutoff by CV: average
'''
RS_train_cf, RS_test_cf = select_RS_by_cv(sigma_set=sigma_kfold,
                                          cv_set=cv_sets,
                                          top_start=top_start,
                                          top_end=top_end,
                                          top_step=top_step,
                                          method='average')

# visualize
sns.heatmap(RS_train_cf.astype(float), cmap='Reds', square=True)
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/cv_performance/pair_cutoff_train_performance_average.pdf', transparent=True)
fig.savefig(path_to_result + output_name + '/cv_performance/pair_cutoff_train_performance_average.png', transparent=True, dpi=300)

# visualize
sns.heatmap(RS_test_cf.astype(float), cmap='Reds', square=True)
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/cv_performance/pair_cutoff_test_performance_average.pdf', transparent=True)
fig.savefig(path_to_result + output_name + '/cv_performance/pair_cutoff_test_performance_average.png', transparent=True, dpi=300)

'''
4. Selection score cutoff by cross-validation: gsea method
'''
# average method
R_train_gsea, R_test_gsea, S_train_gsea, S_test_gsea, RS_train_gsea, RS_test_gsea = select_sig_by_cv(sigma_set=sigma_kfold,
                                                                                                     cv_set=cv_sets,
                                                                                                     top_start=top_start,
                                                                                                     top_end=top_end,
                                                                                                     top_step=top_step,
                                                                                                     method='gsea')

if 'cv_performance' not in os.listdir(path_to_result + output_name + '/'):
    os.mkdir(path_to_result + output_name + '/cv_performance/')

# RS score
visual_cv(score_train=RS_train_gsea,
          score_test=RS_test_gsea,
          title='average CV performance of RS score',
          save_path=path_to_result + output_name + '/cv_performance/RS_score_gsea')
RS_train_gsea.to_csv(path_to_result + output_name + '/cv_performance/RS_train_gsea.csv', sep=',')
RS_test_gsea.to_csv(path_to_result + output_name + '/cv_performance/RS_test_gsea.csv', sep=',')

# R score
visual_cv(score_train=R_train_gsea,
          score_test=R_test_gsea,
          title='average CV performance of R score',
          save_path=path_to_result + output_name + '/cv_performance/R_score_gsea')
R_train_gsea.to_csv(path_to_result + output_name + '/cv_performance/R_train_gsea.csv', sep=',')
R_test_gsea.to_csv(path_to_result + output_name + '/cv_performance/R_test_gsea.csv', sep=',')

# S score
visual_cv(score_train=S_train_gsea,
          score_test=S_test_gsea,
          title='average CV performance of S score',
          save_path=path_to_result + output_name + '/cv_performance/S_score_gsea')
S_train_gsea.to_csv(path_to_result + output_name + '/cv_performance/S_train_gsea.csv', sep=',')
S_test_gsea.to_csv(path_to_result + output_name + '/cv_performance/S_test_gsea.csv', sep=',')

# Performance in validation set
R_val_gsea, S_val_gsea, RS_val_gsea = val_by_select(X_val=X_val,
                                                    y_val=y_val,
                                                    sigma_set=sigma_kfold,
                                                    top_start=top_start,
                                                    top_end=top_end,
                                                    top_step=top_step,
                                                    method='gsea')

R_val_gsea.to_csv(path_to_result + output_name + '/cv_performance/R_val_gsea.csv', sep=',')
S_val_gsea.to_csv(path_to_result + output_name + '/cv_performance/S_val_gsea.csv', sep=',')
RS_val_gsea.to_csv(path_to_result + output_name + '/cv_performance/RS_val_gsea.csv', sep=',')

plt.figure(figsize=(3, 2))
sns.kdeplot(R_val_gsea.mean(axis=1), label='R_in_validation')
sns.kdeplot(S_val_gsea.mean(axis=1), label='S_in_validation')
sns.kdeplot(RS_val_gsea.mean(axis=1), label='RS_in_validation')
plt.tight_layout()
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/cv_performance/validation_performance_gsea.pdf', transparent=True)
fig.savefig(path_to_result + output_name + '/cv_performance/validation_performance_gsea.png', transparent=True, dpi=300)


'''
5. Select R cutoff and S cutoff by CV: gsea
'''
RS_train_cf_gsea, RS_test_cf_gsea = select_RS_by_cv(sigma_set=sigma_kfold,
                                                    cv_set=cv_sets,
                                                    top_start=top_start,
                                                    top_end=top_end,
                                                    top_step=top_step,
                                                    method='gsea')

# visualize
sns.heatmap(RS_train_cf_gsea.astype(float), cmap='Reds', square=True)
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/cv_performance/pair_cutoff_train_performance_gsea.pdf', transparent=True)
fig.savefig(path_to_result + output_name + '/cv_performance/pair_cutoff_train_performance_gsea.png', transparent=True, dpi=300)

# visualize
sns.heatmap(RS_test_cf_gsea.astype(float), cmap='Reds', square=True)
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/cv_performance/pair_cutoff_test_performance_gsea.pdf', transparent=True)
fig.savefig(path_to_result + output_name + '/cv_performance/pair_cutoff_test_performance_gsea.png', transparent=True, dpi=300)

# index is R cutoff, columns is S cutoff
core_sig_idf(sigma_set=sigma_kfold, R_cutoff=0.2, S_cutoff=0.2,
             X_val=X_val, y_val=y_val, validation_type='hold-out',
             do_pathway=False, do_venn=True, save_path=None)

plt.figure(figsize=(3, 2))
sns.kdeplot(RS_test_cf_gsea.max(), label='RS_diff_cutoff')
sns.kdeplot(RS_test_gsea.mean(axis=1), label='RS_same_cutoff')
plt.legend()
plt.tight_layout()
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/cv_performance/diff_same_compare_test_performance_gsea.pdf', transparent=True)
fig.savefig(path_to_result + output_name + '/cv_performance/diff_same_compare_test_performance_gsea.png', transparent=True, dpi=300)


# # 3. Independent validation on MGH set

mgh_home = '/path/to/MGH/set/'

X_mgh = pd.read_csv(mgh_home + 'tige_expr_ave.csv', header=0, index_col=0, sep=',').T
y_mgh = pd.read_csv(mgh_home + 'y_cleaned_R0_NR1.csv', header=0, index_col=0, sep=',')

if 'independent_validation' not in os.listdir(path_to_result + output_name + '/'):
    os.mkdir(path_to_result + output_name + '/independent_validation/')

# Performance in validation set
R_mgh_gsea, S_mgh_gsea, RS_mgh_gsea = val_by_select(X_val=X_mgh,
                                                    y_val=y_mgh,
                                                    sigma_set=sigma_kfold,
                                                    top_start=top_start,
                                                    top_end=top_end,
                                                    top_step=top_step,
                                                    method='gsea')

# Performance in validation set
R_mgh_ave, S_mgh_ave, RS_mgh_ave = val_by_select(X_val=X_mgh,
                                                 y_val=y_mgh,
                                                 sigma_set=sigma_kfold,
                                                 top_start=top_start,
                                                 top_end=top_end,
                                                 top_step=top_step,
                                                 method='average')

plt.figure(figsize=(3, 2))
sns.kdeplot(R_mgh_gsea.mean(axis=1), label='R_in_MGH_validation')
sns.kdeplot(S_mgh_gsea.mean(axis=1), label='S_in_MGH_validation')
sns.kdeplot(RS_mgh_gsea.mean(axis=1), label='RS_in_MGH_validation')
plt.legend()
plt.tight_layout()
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/independent_validation/MGH_performance_gsea.pdf', transparent=True)
fig.savefig(path_to_result + output_name + '/independent_validation/MGH_performance_gsea.png', transparent=True, dpi=300)


plt.figure(figsize=(3, 2))
sns.kdeplot(R_mgh_ave.mean(axis=1), label='R_in_MGH_validation')
sns.kdeplot(S_mgh_ave.mean(axis=1), label='S_in_MGH_validation')
sns.kdeplot(RS_mgh_ave.mean(axis=1), label='RS_in_MGH_validation')
plt.legend()
plt.tight_layout()
fig = plt.gcf()
fig.savefig(path_to_result + output_name + '/independent_validation/MGH_performance_average.pdf', transparent=True)
fig.savefig(path_to_result + output_name + '/independent_validation/MGH_performance_average.png', transparent=True, dpi=300)

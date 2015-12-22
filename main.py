from utils import *
from analysis import *

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering, affinity_propagation
from sklearn.feature_extraction.text import CountVectorizer

def join_comments(s1,s2):
    if(s1 == 'NA'):
        return(s2)
    else:
        if(s2 == 'NA'):
            return(s1)
        else:
            return(s1 + ', ' + s2)

def prepare_data(filenames, bank_names=None, scale_similarities=False, verbose=True):
    all_df = [ pd.read_csv(f,
                    sep=',', encoding='latin1',parse_dates=['Date'], dayfirst=True,
                    ) for f in filenames ]
    if(bank_names is not None):
        for i in range(len(all_df)):
            all_df[i]['Bank'] = bank_names[i]
    df = pd.concat(all_df)
    df.reset_index(inplace=True)
    df.drop('index',axis=1,inplace=True)    

    df['Etiquettes'] = df['Etiquettes'].replace(to_replace=np.nan,value='NA')
    df['Commentaires'] = df['Commentaires'].replace(to_replace=np.nan,value='NA')
    df['Description'] = df.apply(lambda x: join_comments(x['Etiquettes'],x['Commentaires']),axis=1)
    df['Etiquettes_list'] = [e.split(', ') for e in df['Etiquettes']]
    df['Description_list'] = [e.split(', ') for e in df['Description']]

    # Look for every label that appears in the data set
    labels1 = set()
    for v_list in df['Description_list'].values:
        labels1 |= set(v_list)
    labels2 = set()
    for l1 in labels1:
        labels2 |= set(l1.lower().split(' '))
    labels = list(labels2)

    copy_comments = df['Description']
    # change token_pattern to take 1-letter words into account
    textVectorizer = CountVectorizer(vocabulary=labels, stop_words=None, token_pattern="(?u)\\b\\w+\\b")
    bow = textVectorizer.transform(copy_comments)
    if(verbose):
        print 'Vocabulary',textVectorizer.vocabulary_

    dense_bow = bow.todense()
    label_co_occ = np.dot(np.transpose(dense_bow),dense_bow).astype(np.float)
    if(scale_similarities):
        scale = np.copy(np.diag(label_co_occ))
        for i in range(label_co_occ.shape[0]):
            label_co_occ[i,:] = label_co_occ[i,:] / np.sqrt(scale[i])
        for j in range(label_co_occ.shape[0]):
            label_co_occ[:,j] = label_co_occ[:,j] / np.sqrt(scale[j])

    label_tot_expenses = []
    for i in range(len(labels)):
        idx = df.index[np.array(dense_bow[:,i])[:,0] == 1]
        tot_expenses = np.sum(abs(df.loc[idx]['Montant']))
        label_tot_expenses.append(tot_expenses)

    data = {'df': df, 'labels': labels, 'co_occ': label_co_occ,
            'bow_descriptor': dense_bow,
            'labels_inv': textVectorizer.vocabulary_,
            'label_expenses': np.asarray(label_tot_expenses)}
    return(data)


def cluster_details(data,cluster_labels, start_date=None, plot=False, verbose=True):
    labels = data['labels']
    importance = data['label_expenses']

    clusters = []
    nb_cluster = np.max(cluster_labels) +1
    for c in range(nb_cluster):
        if(verbose):
            print 'Cluster',c
        labels_idx = []
        labels_names = []
        for i in range(len(labels)):
            if(cluster_labels[i] == c):
                labels_idx.append(i)
                labels_names.append(labels[i])
        name_idx = np.argmax(importance[labels_idx])
        name = labels[labels_idx[name_idx]]
        print name,':',labels_names
        clusters.append({'idx': labels_idx, 'lab': labels_names, 'name': name})
        if(verbose):
            group_expenses(data,labels_idx, start_date=start_date, plot=plot)

    return(clusters)
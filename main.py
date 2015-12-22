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

def group_indices(data,group_names):
    labels_inv = data['labels_inv']
    group_idx = [labels_inv[label] for label in group_names]
    return(group_idx)

def group_balance(data, group_idx, start_date=None, end_date=None, verbose=True):
    if(verbose):
        print 'Balance for group :', [str(data['labels'][i]) for i in group_idx]

    df = data['df']
    bow_descriptor = np.asarray(data['bow_descriptor'][:,group_idx])
    if(end_date is not None):
        df_idx = df.index[np.any(bow_descriptor,axis=1) & np.asarray(df['Date'] >= start_date) & np.asarray(df['Date'] < end_date)]
    else:
        df_idx = df.index[np.any(bow_descriptor,axis=1) & np.asarray(df['Date'] >= start_date)]        
    group_expenses = df.loc[df_idx]['Montant']

    pos = np.sum(group_expenses[group_expenses>0])
    neg = np.sum(group_expenses[group_expenses<0])

    return([pos,neg,pos+neg])

def group_expenses(data, group_idx, start_date=None, end_date=None, verbose=True, print_balance=True, plot=False):
    if(verbose):
        print 'Expenses for group :', [str(data['labels'][i]) for i in group_idx]

    df = data['df']
    bow_descriptor = np.asarray(data['bow_descriptor'][:,group_idx])
    if(end_date is not None):
        df_idx = df.index[np.any(bow_descriptor,axis=1) & np.asarray(df['Date'] >= start_date) & np.asarray(df['Date'] < end_date)]
    else:
        df_idx = df.index[np.any(bow_descriptor,axis=1) & np.asarray(df['Date'] >= start_date)]        
    group_expenses = df.loc[df_idx]['Montant']

    if(print_balance):
        pos = np.sum(group_expenses[group_expenses>0])
        neg = np.sum(group_expenses[group_expenses<0])
        print 'Gained:', pos, '\tSpent', neg, '\tTotal:', pos+neg

    if(plot):
        group_expenses.plot()
        plt.show()

    return(group_expenses)

def month_to_string(m):
    month = str(m)
    if(m < 10):
        month = '0' + month
    return(month)

def group_monthly_analysis(data, group_idx, start_month=1, end_month=11, fig_args=None):
    res = [group_balance(data,group_idx,
                        start_date='2015-'+month_to_string(m)+'-01',
                        end_date='2015-'+month_to_string(m+1)+'-01',
                        verbose=False)
            for m in range(start_month,end_month+1)]
    res = np.asarray(res)
    # print 'Group :', (' - ').join([str(data['labels'][i]) for i in group_idx])
    bar_width = 0.35

    fig = fig_args['fig']
    if(fig is None):
        ax = plt.figure()
    else:
        ax = fig.add_subplot(fig_args['shape'][0],fig_args['shape'][1],fig_args['sub'])
        ax.set_title(fig_args['title'])
    ax.bar(np.arange(start_month,end_month+1),res[:,2], alpha = 0.5,width = bar_width, color='blue',label='Total')
    ax.bar(np.arange(start_month,end_month+1)+ bar_width,res[:,0], alpha = 0.5,width = bar_width, color='green',label='Gained')
    ax.bar(np.arange(start_month,end_month+1) + 2*bar_width,abs(res[:,1]), alpha = 0.5,width = bar_width, color='red',label='Spent')

    return [np.mean(res[:,0]), np.mean(res[:,1]), np.mean(res[:,2])]

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
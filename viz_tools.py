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

def prepare_data(filename,verbose=True):
    df = pd.read_csv("../../Administratif/Comptes/"+filename,
                       sep=',', encoding='latin1',parse_dates=['Date'], dayfirst=True,
                       #index_col='Date'
                       )

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
    # labels = [str(l) for l in labels]
    if(verbose == True):
        print "Labels:",labels

    copy_comments = df['Description']
    # change token_pattern to take 1-letter words into account
    textVectorizer = CountVectorizer(vocabulary=labels, stop_words=None, token_pattern="(?u)\\b\\w+\\b")
    bow = textVectorizer.transform(copy_comments)
    if(verbose):
        print 'Vocabulary',textVectorizer.vocabulary_

    dense_bow = bow.todense()
    label_co_occ = np.dot(np.transpose(dense_bow),dense_bow)

    data = {'df': df, 'labels': labels, 'co_occ': label_co_occ, 'bow_descriptor': dense_bow, 'labels_inv': textVectorizer.vocabulary_}
    return(data)

def group_indices(data,group_names):
    labels_inv = data['labels_inv']
    group_idx = [labels_inv[label] for label in group_names]
    return(group_idx)

def group_balance(data, group_idx, start_date=None, verbose=True):
    if(verbose):
        print 'Balance for group :', [str(data['labels'][i]) for i in group_idx]

    df = data['df']
    bow_descriptor = np.asarray(data['bow_descriptor'][:,group_idx])
    df_idx = df.index[np.any(bow_descriptor,axis=1) & np.asarray(df['Date'] > start_date)]
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
        df_idx = df.index[np.any(bow_descriptor,axis=1) & np.asarray(df['Date'] > start_date) & np.asarray(df['Date'] < end_date)]
    else:
        df_idx = df.index[np.any(bow_descriptor,axis=1) & np.asarray(df['Date'] > start_date)]        
    group_expenses = df.loc[df_idx]['Montant']

    if(print_balance):
        pos = np.sum(group_expenses[group_expenses>0])
        neg = np.sum(group_expenses[group_expenses<0])
        print 'Gained:', pos, '\tSpent', neg, '\tTotal:', pos+neg

    if(plot):
        group_expenses.plot()
        plt.show()

    return(group_expenses)

def cluster_details(data,cluster_labels, start_date=None, plot=False):
    labels = data['labels']

    clusters = []
    nb_cluster = np.max(cluster_labels) +1
    for c in range(nb_cluster):
        print 'Cluster',c
        labels_idx = []
        labels_names = []
        for i in range(len(labels)):
            if(cluster_labels[i] == c):
                labels_idx.append(i)
                labels_names.append(labels[i])
        clusters.append({'idx': labels_idx, 'lab': labels_names})
        group_expenses(data,labels_idx, start_date=start_date, plot=plot)

def draw_graph(data, verbose=True):
    df = data['df']
    labels = data['labels']
    bow_descriptor = data['bow_descriptor']

    label_tot_expenses = []
    for i in range(len(labels)):
        idx = df.index[np.array(data['bow_descriptor'][:,i])[:,0] == 1]
        tot_expenses = np.sum(df.loc[idx]['Montant'])
        label_tot_expenses.append(tot_expenses)
        # print labels[i],tot_expenses
    net_sizes = np.array([np.sqrt(1+abs(m)) for m in label_tot_expenses])
    net_sizes = 20 + 10.*net_sizes

    # Create network
    label_co_occ = data['co_occ']
    net = nx.from_numpy_matrix(label_co_occ)
    edgelist = net.edges()
    net_labels = dict(zip(range(len(labels)),labels))
    nodelist = range(len(labels))

    # movements = np.zeros((len(labels),len(labels)))
    # edge_w = []
    # for i,j in net.edges():
    #     if(i!=j):
    #         m = np.sum(abs(df[(df[labels[i]] == 1) & (df[labels[j]] == 1)]['Montant']))
    #         m2 = np.sqrt(5+m)
    #         movements[i,j] = m2
    #         movements[j,i] = m2
    #         edge_w.append(0.1*(np.log(10.+m)**2.))
    #     else:
    #         movements[i,j] = 1
    #         edge_w.append(1)
    # edge_w = 0.5*np.asarray(edge_w)

    cluster_labels = affinity_propagation(label_co_occ,max_iter=500,convergence_iter=40)[1]

    plt.figure(figsize=(18,15))
    graph_pos = nx.spring_layout(net)
    nx.draw_networkx_nodes(net, graph_pos, nodelist=nodelist, node_size=net_sizes, node_color=cluster_labels, alpha=0.5)
    nx.draw_networkx_edges(net, edgelist=edgelist, pos=graph_pos)
    nx.draw_networkx_labels(net, graph_pos,labels=net_labels, font_size=12, font_family='sans-serif')
    plt.show()

    return cluster_labels
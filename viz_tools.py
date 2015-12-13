import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering, affinity_propagation
from sklearn.feature_extraction.text import CountVectorizer
from scipy import stats

def join_comments(s1,s2):
    if(s1 == 'NA'):
        return(s2)
    else:
        if(s2 == 'NA'):
            return(s1)
        else:
            return(s1 + ', ' + s2)

def prepare_data(filename):
    data = pd.read_csv("../../Administratif/Comptes/"+filename,
                       sep=',', encoding='latin1',parse_dates=['Date'], dayfirst=True,
                       #index_col='Date'
                       )

    data['Etiquettes'] = data['Etiquettes'].replace(to_replace=np.nan,value='NA')
    data['Commentaires'] = data['Commentaires'].replace(to_replace=np.nan,value='NA')
    data['Description'] = data.apply(lambda x: join_comments(x['Etiquettes'],x['Commentaires']),axis=1)
    data['Etiquettes_list'] = [e.split(', ') for e in data['Etiquettes']]
    data['Description_list'] = [e.split(', ') for e in data['Description']]

    return(data)

def draw_graph(data, print_clusters=True):
    N = data.shape[0]

    labels = set()
    for v_list in data['Etiquettes_list'].values:
        labels |= set(v_list)
    labels = list(labels)
    # print labels

    for l in labels:
        data[l]= [1 if l in v else 0  for v in data['Etiquettes_list']]

    tot_montant = [np.sum(data[data[l] == 1]['Montant']) for l in labels]
    tot_montant
    net_colors = ['r' if m < 0 else 'g' for m in tot_montant]
    net_sizes = np.array([np.sqrt(1+abs(m)) for m in tot_montant])
    net_sizes = 50 + 5.*net_sizes

    df = pd.DataFrame(index=labels, columns=labels).fillna(0)
    for l in data['Etiquettes_list']:
        for e1 in l:
            for e2 in l:
                df[e1][e2] += 1
                df[e2][e1] += 1

    # Create network
    net = nx.from_numpy_matrix(df.values)
    idx = df.index.values
    net_labels = dict(zip(range(idx.shape[0]),idx.T))
    edgelist = net.edges()

    colabels = df.values
    for i in range(len(labels)):
        colabels[i,i] = 2

    movements = np.zeros((len(labels),len(labels)))
    edge_w = []
    edge_color = []
    for i,j in net.edges():
        if(i!=j):
            m = np.sum(abs(data[(data[labels[i]] == 1) & (data[labels[j]] == 1)]['Montant']))
            m2 = np.sqrt(5+m)
            movements[i,j] = m2
            movements[j,i] = m2
            edge_w.append(0.1*(np.log(10.+m)**2.))
        else:
            movements[i,j] = 1
            edge_w.append(1)
        edge_color.append('black')

    copy_comments = list(data['Commentaires'].replace(to_replace=np.nan,value='NA'))
    textVectorizer = CountVectorizer(ngram_range=(1,2))
    bow = textVectorizer.fit_transform(copy_comments)

    dense_bow = bow.todense()
    covariances = np.cov(dense_bow)

    if(False):
        # Take a look at the covariances
        print np.max(covariances)
        print np.min(covariances)
    min_cov = np.min(covariances)

    idx_labels = {}
    for i in range(len(labels)):
        idx_labels[labels[i]] = i
    # print idx_labels
    affinity = np.zeros((len(labels),len(labels)))
    for l in range(len(labels)):
        rows = [i for i in range(N) if (data[labels[l]][i] == 1)]
        for r in rows:
            if not (copy_comments[r] == 'NA'):
                for j in range(N):
                    cov = 100.*(min_cov+covariances[r,j])
                    if(cov != 0):
                        for ll in data['Etiquettes_list'][j]:
                            a = max(cov,0)
                            affinity[l,idx_labels[ll]] += a
                            affinity[idx_labels[ll],l] += a

    # Add affinity edges from comments similiraity
    cov_threshold = 10.
    for i in range(len(labels)):
        for j in range(i):
            if(affinity[i,j] > cov_threshold):
                if not(net.has_edge(i,j)):
                    net.add_edge(i,j)
                    edgelist.append((i,j))
                    # print i,j
                    edge_color.append('red')
                    edge_w.append(3.+100.*covariances[i,j])

    affinity = affinity/np.std(affinity)
    movements = movements/np.std(movements)
    colabels = colabels /np.std(colabels)
    complete_affinity = affinity + 2.*colabels

    cluster_labels = affinity_propagation(complete_affinity,max_iter=500,convergence_iter=40)[1]
    plt.figure(figsize=(18,15))

    graph_pos = nx.spring_layout(net)

    nx.draw_networkx_nodes(net, graph_pos, node_size=net_sizes, node_color=cluster_labels, alpha=0.5)
    nx.draw_networkx_edges(net, edgelist =edgelist, pos=graph_pos, edge_color=edge_color,width=0.5*np.asarray(edge_w))
    nx.draw_networkx_labels(net, graph_pos,labels = net_labels, font_size=12, font_family='sans-serif')

    plt.show()

    if(print_clusters):
        nb_cluster = np.max(cluster_labels) +1
        for c in range(nb_cluster):
            print('Cluster',c)
            print([labels[i] for i in range(len(labels)) if ( cluster_labels[i] == c) ])

    return bow,affinity
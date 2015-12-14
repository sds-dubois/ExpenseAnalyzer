import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering, affinity_propagation
from sklearn.feature_extraction.text import CountVectorizer

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
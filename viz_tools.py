import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering, affinity_propagation
from sklearn.feature_extraction.text import CountVectorizer

def draw_graph(data, cluster_preference=True, verbose=True):
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

    # Create network
    label_co_occ = data['co_occ']
    net = nx.from_numpy_matrix(label_co_occ)
    edgelist = net.edges()
    net_labels = dict(zip(range(len(labels)),labels))
    nodelist = range(len(labels))

    if(cluster_preference):
        M = np.percentile(label_co_occ,75.)
        m = np.min(label_co_occ)
        sizeM = np.percentile(net_sizes,75.)
        preference = m + np.asarray([min(s,sizeM) for s in net_sizes])*((M-m)/sizeM)
        cluster_labels = affinity_propagation(label_co_occ,preference=preference, max_iter=500,convergence_iter=40)[1]        
    else:
        cluster_labels = affinity_propagation(label_co_occ,max_iter=500,convergence_iter=40)[1]

    plt.figure(figsize=(18,15))
    graph_pos = nx.spring_layout(net)
    nx.draw_networkx_nodes(net, graph_pos, nodelist=nodelist, node_size=(20 + 10.*net_sizes), node_color=cluster_labels, alpha=0.5)
    nx.draw_networkx_edges(net, edgelist=edgelist, pos=graph_pos)
    nx.draw_networkx_labels(net, graph_pos,labels=net_labels, font_size=12, font_family='sans-serif')
    plt.show()

    return cluster_labels
from config import *

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering, affinity_propagation
from sklearn.feature_extraction.text import CountVectorizer

def draw_graph(data, g, cluster_labels):
    labels = data['labels']
    net_sizes = np.array([np.sqrt(1+m) for m in data['label_expenses']])
    edgelist = g.edges()
    net_labels = dict(zip(range(len(labels)),labels))
    nodelist = range(len(labels))

    plt.figure(figsize=(18,15))
    graph_pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, graph_pos, nodelist=nodelist, node_size=(20 + 10.*net_sizes),
                            node_color=np.asarray(all_colors)[cluster_labels], alpha=0.5)
    nx.draw_networkx_edges(g, edgelist=edgelist, pos=graph_pos)
    nx.draw_networkx_labels(g, graph_pos,labels=net_labels, font_size=12, font_family='sans-serif')
    plt.show()


def cluster_keywords(data,min_size=2, cluster_preference=True, verbose=True):
    start_cluster_idx = 0
    clusters = np.zeros(len(data['labels']), dtype=int)

    # create graph
    label_co_occ = data['co_occ']
    g = nx.from_numpy_matrix(label_co_occ)

    if(cluster_preference):
        # define preferences
        net_sizes = np.array([np.sqrt(1+m) for m in data['label_expenses']])
        M = np.percentile(label_co_occ,75.)
        m = np.min(label_co_occ)
        sizeM = np.percentile(net_sizes,75.)
        preference = m + np.asarray([min(s,sizeM) for s in net_sizes])*((M-m)/sizeM)

    for comp in sorted(nx.connected_components(g), key=len, reverse=True):
        l = len(comp)
        if(l >= min_size):
            temp_co_occ = (label_co_occ[:,comp])[comp,:]
            if(cluster_preference):
                [n_clusters, temp_clusters] = affinity_propagation(temp_co_occ,
                                                                    preference=preference[comp],
                                                                    max_iter=500,
                                                                    convergence_iter=40)
            else:
                [n_clusters, temp_clusters] = affinity_propagation(temp_co_occ,
                                                                    max_iter=500,
                                                                    convergence_iter=40)
            for i in xrange(l):
                clusters[comp[i]] = temp_clusters[i] + start_cluster_idx
            start_cluster_idx += len(n_clusters)
            if(verbose):
                print('Found component of size ' + str(l) + ' and added ' \
                        + str(len(n_clusters)) + ' clusters')
        else:
            # only one cluster for this component
            if(verbose):
                print('Found component of size ' + str(l) + ' so do not run affinity_propagation')
            for n in comp:
                clusters[n] = start_cluster_idx            
            start_cluster_idx += 1

    return g, clusters
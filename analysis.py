from utils import *

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

from data.extractor import *
from data.ListSnP500 import *
from Nosputin.utility.utility import *

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn
import plotly.graph_objects as go

from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def get_latent_value(values, method='kernelpca', normalization=True, widget_key=None):
    st.write("Dimensionality reduction for dim: <{}*{}>".format(len(values), len(values[0])))
    
    if normalization:
        values = normalize(values, norm='l2')
    
    if method=='pca':
        pca = PCA(n_components=2,whiten=True)
        pca.fit(np.transpose(values))
        return pca.components_
    
    elif method=='sparsepca':
        sparse_pca = SparsePCA(n_components=2)
        return np.transpose(sparse_pca.fit_transform(values))
    
    elif method == 'kernelpca':
        kernel_pca = KernelPCA(n_components=2, kernel='rbf')
        return np.transpose(kernel_pca.fit_transform(values))
    
    elif method == 'tsne':
        n_it = st.slider("Max iteration", min_value=5000, max_value=50000, key='tsne_it_{}'.format(widget_key))
        perp = st.slider("Perplexity", min_value=30, max_value=300, key='tsne_prep{}'.format(widget_key))
        lr = st.slider("Learning rate", min_value=10, max_value=1000, key='tsne_lr{}'.format(widget_key))
        tsne = TSNE(n_components=2, n_iter=n_it, perplexity=perp, learning_rate=lr, n_jobs=4)
        return np.transpose(tsne.fit_transform(values))


def latent_view(dataset):
    timelines = dataset.timePeriods
    
    lengths = show_dist(dataset)
    
    sel_dim = st.selectbox("Number of reported features select", sorted(list(set(lengths)), reverse=True))
     
    selected_values, selected_tickers, performance_values = dataset.get_raw_value(sel_dim= int(sel_dim))
    st.write("{} companies total ".format(len(selected_values)))
    st.write("If you use normalization, the scale of each company will affect less.")
    
    is_norm = st.checkbox("Normalization", value=True)
    
    st.write("If you want to eliminate uncertainty of small companies, you can choose using only S&P 500 companies")
    snp_only = st.checkbox("S&P 500 only", value=False)
    
    if snp_only:
        selected_values, selected_tickers, performance_values = snp500_filter(selected_values, selected_tickers, performance_values)
    
    sel_method = st.selectbox("projection method",['pca','kernelpca','sparsepca','tsne'])
    st.write("We highly recommend to use t-sne in normal.")
    try:
        latent_values = get_latent_value(selected_values, method=sel_method, normalization=is_norm)
        st.write(latent_values)
        
        show_plotly(latent_values, performance_values, selected_tickers, name="selected companies", max_y=200, min_y=-200)
        
    except ValueError:
        st.write("Please select the number of reported feature except for 0.")
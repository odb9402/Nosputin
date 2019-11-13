from data.extractor import *
from data.ListSnP500 import *
from Nosputin.utility.utility import *
from Nosputin.latent_view import *
from Nosputin.gpModel.defineModel import *
from sklearn.preprocessing import normalize

import plotly
import plotly.figure_factory as ff
import tensorflow as tf
import pandas as pd
import numpy as np
import gpflow
import streamlit as st
import random
import math
from scipy import stats

class NosputinModelBuilder():
    def __init__(self, dataset, sel_dim, sep_date, snp_only=False):
        self.dataset = dataset
        self.model = None
        self.x = [] 
        self.y = []
        self.ticker = []
        self.x_t = []
        self.y_t = []
        self.ticker_t = []
        self.dim_x_ = sel_dim
        
        self.x, self.ticker, self.y = self.dataset.get_raw_value(self.dim_x_, sep_date)
        if snp_only:
            self.x, self.ticker, self.y = snp500_filter(self.x, self.ticker, self.y)
        self.dimension_filter()
        self.train_test_split()
    
    def train_nosputin_model(self, model='deepKernel', norm=False):
        gpflow.reset_default_graph_and_session()

        xdim = len(self.x[0])
        n_data = len(self.x)
        np_x = np.array(self.x)
        np_y = np.array(self.y, dtype=np.float64).reshape(len(self.y),1)

        st.write(np_x)

        st.write("Feature dimension of company : {}".format(xdim))
        st.write("dim(X):{} , dim(Y):{}".format(np_x.shape, np_y.shape))
        
        if model == 'naive':
            kernel = gpflow.kernels.Matern52(input_dim=xdim)
            self.model = gpflow.models.GPR(np_x, np_y, kern=kernel, mean_function=None)
            naive_opt = gpflow.train.AdamOptimizer().minimize(self.model)
            
        elif model == 'deepKernel':
            float_type = gpflow.settings.float_type
            ITERATIONS = notebook_niter(5000)
            minibatch_size = notebook_niter(5000, test_n=10)

            global is_test
            is_test = False

            nn_dim = 2 
            nn_function = lambda x: tf.cast(fully_nn_layer2(tf.cast(x, tf.float32), xdim, nn_dim), float_type) 
            base_kernel = SMKernel(5,nn_dim)
            M = 50
            Z = np_x[:M,:].copy()
            deep_kernel = NNComposedKernel(base_kernel, nn_function)
            self.model = NN_SGPR(np_x, np_y, Z=Z, kern=deep_kernel)

            opt = gpflow.train.AdamOptimizer().minimize(self.model, maxiter=ITERATIONS)

            is_test=True
        elif model == 'sm':
            M = 50 # Number of inducing points
            kernel = SMKernel(5, xdim)
            Z = np_x[:M,:].copy()
            self.model = gpflow.models.SGPR(np_x, np_y, Z=Z, kern=kernel, mean_function=None)
            naive_opt = gpflow.train.AdamOptimizer().minimize(self.model)
            
        else:
            pass
        st.write(self.model.as_pandas_table())

    
    def train_test_split(self, test_ratio=0.2):
        raw_len = len(self.x)
        while True:
            if len(self.x)/raw_len <= 1 - test_ratio:
                break
            i = random.randint(0, len(self.x)-1)
            self.x_t.append(self.x.pop(i))
            self.y_t.append(self.y.pop(i))
            self.ticker_t.append(self.ticker.pop(i))
            
    def dimension_filter(self):
        dims = []
        for x_ in self.x:
            dims.append(len(x_))
        dims = list(set(dims))
        counts = dict(zip(dims, [ 0 for x in range(len(dims)) ]))
        for d in dims:
            for x_ in self.x:
                if len(x_) == d:
                    counts[d] = counts[d] + 1
        freq_dims = max(counts, key=(lambda k: counts[k]))
        i = 0
        while True:
            if i == len(self.x) - 1:
                break
            if len(self.x[i]) != freq_dims:
                self.x.pop(i)
                self.y.pop(i)
                self.ticker.pop(i)
            else:
                i = i + 1
    
    def save_model(self, name=""):
        saver = gpflow.saver.Saver()
        saver.save(name, self.model)
        
    def load_model(self, name=""):
        with tf.Graph().as_default() as graph, tf.Session().as_default():
            self.model = saver.load(name)

    def show_plotly_result(self):
        pass

def show_rmse(preds, real):
    mse = 0.0
    preds=preds.flatten()
    for x,y in zip(preds, real):
        mse += (x-y)**2
    return math.sqrt(mse/len(real))


def get_errors(preds, real):
    errors = []
    preds=preds.flatten()
    for x, y in zip(preds, real):
        errors.append(abs(x-y))
    return errors


def build_nosputin_model(dataset):
    timelines = dataset.timePeriods
    
    st.write("Build nosputin model based on Gaussian process model on test")
    
    sep_date = str(st.date_input("Date partition", value=datetime(2015,1,1)))
    st.write("In order to train futrue performances of each stock, Nosputin dataset should be seperated into two parts.")
    
    lengths = show_dist(dataset, sep_date=None)#sep_date)
    sel_dim = st.selectbox("Number of reported features select", sorted(list(set(lengths)), reverse=True))
     
    st.write("If you want to eliminate uncertainty of small companies, you can choose using only S&P 500 companies")
    snp_only = st.checkbox("S&P 500 only", value=False)
    
    model_builder = NosputinModelBuilder(dataset, sel_dim, sep_date, snp_only)
    
    st.write("{} companies total for training".format(len(model_builder.x)))
    st.write("{} companies total for testing".format(len(model_builder.x_t)))
    is_norm = st.checkbox("Normalize", value=False)
    
    if is_norm:
        model_builder.x = normalize(model_builder.x)
        model_builder.x_t = normalize(model_builder.x_t)
    
    st.subheader("Gaussian process regression")
    
    model_builder.train_nosputin_model()
    
    train_mean, train_var = model_builder.model.predict_f(np.array(model_builder.x)) 
    test_mean, test_var = model_builder.model.predict_f(np.array(model_builder.x_t))
    
    st.write("RMSE for training data : {}".format(show_rmse(train_mean, model_builder.y)))
    st.write("RMSE for test data : {}".format(show_rmse(test_mean, model_builder.y_t)))
    
    st.subheader("Prediction results projection")
    
    sel_method = st.selectbox("projection method",['tsne','kernelpca','pca','sparsepca'])
    
    latent_values = get_latent_value(model_builder.x, method=sel_method, normalization=False, widget_key='1')
    latent_values_test = get_latent_value(model_builder.x_t, method=sel_method, normalization=False, widget_key='2')
    train_errors = get_errors(train_mean, model_builder.y)
    test_errors = get_errors(test_mean, model_builder.y_t)
    
    show_plotly(latent_values, train_errors, model_builder.ticker, 'train errors')
    show_plotly(latent_values_test, test_errors, model_builder.ticker_t, 'test errors')
    show_plotly(latent_values_test, test_var.flatten(), model_builder.ticker_t, 'test uncertainties') 
    
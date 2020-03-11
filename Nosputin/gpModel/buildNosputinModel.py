from data.extractor import *
from data.ListSnP500 import *
from Nosputin.utility.utility import *
from Nosputin.latent_view import *
from Nosputin.gpModel.defineModel import *
from sklearn.preprocessing import normalize

import itertools
import glob
import plotly
import plotly.figure_factory as ff
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as tfkl
import pandas as pd
import numpy as np
#import gpflow
import streamlit as st
import random
import math

from scipy import stats
from array import array
from copy import deepcopy

class NosputinModelBuilder():
    def __init__(self, dataset):#, sel_dim, sep_date, snp_only=False):
        self.dataset = dataset
        self.x = []
        self.y = []
        self.ticker = []
        self.noises = []

        self.x_t = []
        self.y_t = []
        self.ticker_t = []
        self.noises_t = []
        
    def model_build(self):
        self.model = MassiveBayesianNNNosputinModel(self.x, self.y, self.noises, self.x_t, self.y_t, self.noises_t) 
    
    def load_data(self, sel_dim, sep_date, snp_only=False):
        self.x, self.ticker, self.y, self.noises = self.dataset.get_raw_value(sel_dim, sep_date, noise_call=True)
        if snp_only:
            self.x, self.ticker, self.y, self.noises = snp500_filter(self.x, self.ticker, self.y, self.noises)
        self.dimension_filter()
    
    def snp500_filter(self):
        self.x, self.ticker, self.y, self.noises = snp500_filter(list(self.x), self.ticker, self.y, self.noises)
        
    def train_test_split(self, test_ratio=0.2):
        raw_len = len(self.x)
        while True:
            if len(self.x)/raw_len <= 1 - test_ratio:
                break
            i = random.randint(0, len(self.x)-1)
            self.x_t.append(self.x.pop(i))
            self.y_t.append(self.y.pop(i))
            self.ticker_t.append(self.ticker.pop(i))
            self.noises_t.append(self.noises.pop(i))

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

    def show_plotly_result(self):
        pass

    def export_data(self, export_dir="./data/training_set/"):
        num_c = len(self.x)

        prog = st.progress(0)
        j = 0

        header = self.dataset.getCompany(self.ticker[0])
        date = str(header.data.index[0]).split()[0]

        for i in range(num_c):
            np_file = open(export_dir + self.ticker[i] + "_" + date + ".npd","wb")
            np_list = deepcopy(self.x[i])
            np_list.append(self.noises[i])
            np_list.append(self.y[i])

            np_arr = array('d', np_list)
            np_arr.tofile(np_file)
            np_file.close()
            j += 1
            prog.progress(int(j/num_c*100))
        st.write("Export data complete.")

    def import_data(self, import_dir="./data/training_set/*"):
        file_names = glob.glob(import_dir)

        prog = st.progress(0)
        i = 0
        for file_name in file_names:
            
            #if file_name.rsplit('.',1)[0].rsplit('_',1)[1].split('-')[0] == '2014':
            #    pass
            #else:
            #    continue
            
            data_file = open(file_name, 'rb')
            data_arr = array('d')
            data_arr.fromstring(data_file.read())
            data_list = list(data_arr)
            
            Y = data_list.pop()
            noise = data_list.pop()
            
            if Y > 300:
                continue
            elif Y < -90:
                continue
            self.y.append(Y)
            self.noises.append(noise)
            self.x.append(data_list)
            self.ticker.append(file_name.rsplit('/',1)[1])
            i += 1
            prog.progress(int(i/len(file_names)*100))

def show_rmse(preds, real):
    mse = 0.0
    #preds=preds.flatten()
    for x,y in zip(preds, real):
        mse += (x-y)**2
    return math.sqrt(mse/len(real))


def get_errors(preds, real):
    errors = []
    #preds=preds.flatten()
    for x, y in zip(preds, real):
        errors.append(abs(x-y))
    return errors


def build_nosputin_model(dataset):
    timelines = dataset.timePeriods

    st.write("Build nosputin model based on Gaussian process model on test")

    st.write("If you want to eliminate uncertainty of small companies, you can choose using only S&P 500 companies")
    snp_only = st.checkbox("S&P 500 only", value=False)
    is_norm = st.checkbox("Normalize", value=False)
    
    model_builder = NosputinModelBuilder(dataset)
    model_builder.import_data()
    if snp_only:
        model_builder.snp500_filter()
    model_builder.train_test_split()
    model_builder.model_build()

    st.write("{} companies total for training".format(len(model_builder.x)))
    st.write("{} companies total for testing".format(len(model_builder.x_t)))

    if is_norm:
        model_builder.x = normalize(model_builder.x)
        model_builder.x_t = normalize(model_builder.x_t)
    
    st.subheader("Nosputin Model Training")
    st.write(model_builder.model.model)

    if st.button("Model train"):
        model = model_builder.model
        model.train_model()
        #train_mean, train_var = model_builder.model.predict_f(np.array(model_builder.x))
        #test_mean, test_var = model_builder.model.predict_f(np.array(model_builder.x_t))
        
        train_mean, test_mean = model.get_preds()
        
        st.write("RMSE for training data : {}".format(show_rmse(train_mean, model_builder.y)))
        st.write("RMSE for test data : {}".format(show_rmse(test_mean, model_builder.y_t)))

        st.subheader("Prediction results projection")

        sel_method = st.selectbox("projection method",['pca','kernelpca','sparsepca','tsne'])

        latent_values = get_latent_value(model_builder.x, method=sel_method, normalization=False, widget_key='1')
        latent_values_test = get_latent_value(model_builder.x_t, method=sel_method, normalization=False, widget_key='2')
        train_errors = get_errors(train_mean, model_builder.y)
        test_errors = get_errors(test_mean, model_builder.y_t)

        #show_plotly(latent_values, train_errors, model_builder.ticker, 'train errors')
        #show_plotly(latent_values_test, test_errors, model_builder.ticker_t, 'test errors')
        #show_plotly(latent_values, train_var.flatten(), model_builder.ticker, 'test uncertainties')
        #show_plotly(latent_values_test, test_var.flatten(), model_builder.ticker_t, 'test uncertainties')
        
        show_plotly(latent_values, model_builder.y, model_builder.ticker, 'train ground truth')
        show_plotly(latent_values, train_mean, model_builder.ticker,'train prediction')
        
        show_plotly(latent_values_test, model_builder.y_t, model_builder.ticker_t, 'test ground truth')
        show_plotly(latent_values_test, test_mean, model_builder.ticker_t,'test prediction')
    else:
        pass


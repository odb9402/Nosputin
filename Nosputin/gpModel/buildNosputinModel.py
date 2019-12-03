from data.extractor import *
from data.ListSnP500 import *
from Nosputin.utility.utility import *
from Nosputin.latent_view import *
from Nosputin.gpModel.defineModel import *
from sklearn.preprocessing import normalize

import glob
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
from array import array
from copy import deepcopy

class NosputinModelBuilder():
    def __init__(self, dataset):#, sel_dim, sep_date, snp_only=False):
        self.dataset = dataset
        self.model = None
        self.x = []
        self.y = []
        self.ticker = []
        self.noises = []

        self.x_t = []
        self.y_t = []
        self.ticker_t = []
        self.noises_t = []

    def load_data(self, sel_dim, sep_date, snp_only=False):
        self.x, self.ticker, self.y, self.noises = self.dataset.get_raw_value(sel_dim, sep_date, noise_call=True)
        if snp_only:
            self.x, self.ticker, self.y, self.noises = snp500_filter(self.x, self.ticker, self.y, self.noises)
        self.dimension_filter()
        #self.train_test_split()

    def train_nosputin_model(self, model='deepKernel', norm=False):
        with st.spinner("Reset gpflow graph. . ."):
            gpflow.reset_default_graph_and_session()

        st.write("Reset gpflow graph. . . ")
        xdim = len(self.x[0])
        n_data = len(self.x)
        np_x = np.array(self.x)
        y_n_noises = [self.y, self.noises]
        np_y = np.array(self.y, dtype=np.float64).reshape(len(self.y),1)
        np_y_n_noises = np.array(y_n_noises, dtype=np.float64).reshape(len(self.y),2)

        st.write("Feature dimension of company : {}".format(xdim))
        st.write("dim(X):{} , dim(Y):{}".format(np_x.shape, np_y.shape))
        
        if model == 'naive':
            kernel = gpflow.kernels.Matern52(input_dim=xdim)
            self.model = gpflow.models.GPR(np_x, np_y, kern=kernel, mean_function=None)
            naive_opt = gpflow.train.AdamOptimizer().minimize(self.model)

        elif model == 'deepKernel':
            float_type = gpflow.settings.float_type
            ITERATIONS = notebook_niter(10000)
            minibatch_size = notebook_niter(10000, test_n=10)

            global is_test
            is_test = False

            nn_dim = 2
            nn_function = lambda x: tf.cast(fully_nn_layer(tf.cast(x, tf.float32), xdim, nn_dim), float_type)
            base_kernel = SMKernel(5,nn_dim)
            M = 50
            Z = np_x[:M,:].copy()

            likelihood = HeteroskedasticGaussian()

            deep_kernel = NNComposedKernel(base_kernel, nn_function)
            self.model = NN_SVGP(np_x, np_y_n_noises, Z=Z, kern=deep_kernel, likelihood=likelihood)
            with st.spinner('GPflow training process. . . .'):
                opt = gpflow.train.AdamOptimizer().minimize(self.model, maxiter=ITERATIONS)
            st.success('Done.')
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


    def train_test_split(self, test_ratio=0.3):
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

    def save_model(self, name=""):
        saver = gpflow.saver.Saver()
        saver.save(name, self.model)

    def load_model(self, name=""):
        with tf.Graph().as_default() as graph, tf.Session().as_default():
            self.model = saver.load(name)

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
            data_file = open(file_name, 'rb')
            data_arr = array('d')
            data_arr.fromstring(data_file.read())
            data_list = list(data_arr)
            
            Y = data_list.pop()
            noise = data_list.pop()
            
            self.y.append(Y)
            self.noises.append(noise)
            self.x.append(data_list)
            self.ticker.append(file_name.rsplit('.')[0].rsplit('_')[0])
            i += 1
            prog.progress(int(i/len(file_names)*100))

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

    st.write("If you want to eliminate uncertainty of small companies, you can choose using only S&P 500 companies")
    snp_only = st.checkbox("S&P 500 only", value=False)
    is_norm = st.checkbox("Normalize", value=False)

    model_builder = NosputinModelBuilder(dataset)
    model_builder.import_data()
    model_builder.train_test_split()

    st.write("{} companies total for training".format(len(model_builder.x)))
    st.write("{} companies total for testing".format(len(model_builder.x_t)))

    if is_norm:
        model_builder.x = normalize(model_builder.x)
        model_builder.x_t = normalize(model_builder.x_t)

    st.subheader("Gaussian process regression")

    if st.button("Model train"):
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
    else:
        pass


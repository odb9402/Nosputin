from data.extractor import *
import numpy as np
import streamlit as st
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from PIL import Image
import pymongo
import tensorflow as tf
import gpflow
plt.style.use('ggplot')

from Nosputin.companyDataset import *
from Nosputin.data_browser import data_browser 
from Nosputin.latent_view import latent_view
from Nosputin.gpModel.buildNosputinModel import build_nosputin_model

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def show_main():
    st.markdown("Nosputin is an integrated tool to analyze stock market data for each company with its fundamental data such as revenues and net income.")
    st.markdown("## Description")
    st.markdown("**Nosputin** gives 4 application modes including **Data browser, Latent-View, Build model** and **Prediction**. After select any mode of **Nosputin, Nosputin** will load the stock market database from **Nosputin** database. You can adjust the range of the date of data using the sidebar.")
    st.markdown("### Data browser")
    st.markdown("**The Data browser** model will show the fundamental information for the selected individual stock. You can select a stock using a ticker of the stock. ")
    browser_image = Image.open("./pic/databrowser.png")
    st.image(browser_image, use_column_width=True)
    st.markdown("### Latent-view")
    st.markdown("**Latent-view** can show a 2-dimensional projection of fundamental data for each company using dimensionality reduction or manifold-learning algorithms. It allows the visual inspection that can check the distribution and similarity of the companies in terms of their financial statements such as revenue.")
    st.markdown("Note that since the dimension, the number of reported fundamental data has to be the same to show latent space of data. Therefore, we highly recommend that the date range to load stock market data should be as XXXX.01.01 to YYYY.12.31.")
    st.markdown("If you select the dimension of data, **Nosputin** will show the scatter plot with a 2-dimensional latent view and the performance during the selected date range.")
    latent_image = Image.open("./pic/example_latentview.png")
    st.image(latent_image, use_column_width=True)
    st.markdown("### Build model")
    st.markdown("**Build model** will create the model that can predict the performance for each stock data using the machine learning-based implementation. This version uses Gaussian process to infer the performance from their fundamental data, that is, it has a huge advantage of verifying uncertainty of the prediction not only predicting the performance itself. This advantage that can measure the uncertainty of prediction is much more important in financial inference.")
    st.markdown(" Like **Latent-view**, the stock-prediction model only can predict the data that has the same dimensions because of the limitation of Gaussian process like any other machine learning-based technology. ")
    st.markdown("### Prediction")
    

@st.cache
def load_data(file, start_date, end_date):
    return SimFinDataset(file,'comma', start_date, end_date,True)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data_mongo():
    base_db_client = pymongo.MongoClient('dmblab.ml', 27017,
                                        username='nosputin_reader',
                                        password='nosputin',
                                        authSource='Nosputin_db')
    nosputin_db_client = base_db_client['Nosputin_db']
    
    dataset = CompanyDataset()
    dataset.loadData_mongo(nosputin_db_client)
    return dataset

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def select_data_by_date(dataset, start_date, end_date):
    prog = st.progress(0)
    i = 0
    for c in dataset.companies:
        i += 1
        c.select_data_by_date(start_date=start_date, end_date=end_date)
        prog.progress(int(i/len(dataset.companies)*100))
    return dataset


def main():
    st.title('Nosputin Beta')
    st.balloons()
    
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                   ["Main page","Data browser", "Latent-View", "Build model", "Prediction"])
    st.sidebar.subheader("Select date range")
    start_date = st.sidebar.date_input("START DATE", value=datetime(2012,12,31))
    end_date = st.sidebar.date_input("END DATE", value=datetime(2015,12,31))
    #dataset = load_data('data/output-comma-wide-tec.csv',str(start_date), str(end_date))
    
    if app_mode =="Main page":
        show_main()
    
    dataset = load_data_mongo()
    if app_mode == "Data browser":
        data_browser(dataset)
    elif app_mode == "Latent-View":
        dataset = select_data_by_date(dataset, str(start_date), str(end_date))
        latent_view(dataset)
    elif app_mode == "Build model":
        dataset = select_data_by_date(dataset, str(start_date), str(end_date))
        build_nosputin_model(dataset)
    elif app_mode == "Prediction":
        pass
    
if __name__ == '__main__':
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)               # The logger object only output logs which have
                                                # upper level than INFO.
    log_format = logging.Formatter('%(asctime)s:%(message)s')

    stream_handler = logging.StreamHandler()    # Log output setting for the command line.
    stream_handler.setFormatter(log_format)     # The format of stream log will follow this format.
    logger.addHandler(stream_handler)

    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("i hope you complete. \n")
        sys.exit()

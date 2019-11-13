import pandas as pd
from data.extractor import *
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from Nosputin.companyDataset import *

def show_data(ticker, dataset, timelines=None):
    st.subheader('Fundamental data of {}'.format(ticker))
    pd_company_data = dataset.getCompany(ticker).data.dropna(how='any')
    
    st.write(pd_company_data)
    indi_show = None
    st.subheader('Flow chart of : {}'.format(ticker))
    indi_show = st.selectbox("Indicator", pd_company_data.columns)
    st.write('Industry : {}'.format(dataset.getCompany(ticker).industry))
    try:
        fig, ax = plt.subplots(figsize=(14,10)
                              ,dpi=300)
        ax = plt.plot(pd_company_data[str(indi_show)]
                     ,linewidth=6
                     ,color='c')
        plt.yticks(fontsize=16)
        plt.xticks(rotation='40', fontsize=16)
        plt.title("{} : {}".format(ticker,str(indi_show)), fontsize=30)
        st.pyplot(dpi=300)
    except:
        st.write("Cannot find the ticker {}".format(ticker))
        st.write("Use another timeline or ticker")

def show_raw_dist(dataset):
    pass
        
def data_browser(dataset):
    timelines = dataset.timePeriods
    st.write("Total number of companies : {}".format(dataset.numCompanies))
    st.write("Total number of indicators : {}".format(dataset.numIndicators))
    
    ticker_ex = st.selectbox("LIST OF TICKERS", sorted(dataset.tickers))
    
    ticker_text = st.text_input("SELECT TICKER : ", value=ticker_ex)
    show_data(ticker_text, dataset, timelines)

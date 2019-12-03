import pandas as pd
from data.extractor import *
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.style.use('ggplot')

from Nosputin.companyDataset import *

def show_data(ticker, dataset, timelines=None):
    st.subheader('Fundamental data of {}'.format(ticker))
    company = dataset.getCompany(ticker)
    pd_company_data = company.data
    pd_prices = company.prices
    indi_show = None
    
    st.write(pd_company_data)
    st.subheader('Flow chart of : {}'.format(ticker))
    indi_show = st.selectbox("Indicator", pd_company_data.columns)
    st.write('Industry : {}'.format(dataset.getCompany(ticker).industry))
    
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=pd_prices.index, y=pd_prices.values, name="Prices of {}".format(indi_show), line_color='deepskyblue'))

    fig_price.update_layout(title_text="Prices of {}".format(indi_show),
                          xaxis_rangeslider_visible=True,
                          margin=dict(l=0,r=0,b=0))
    st.write("Standard variation of the price of {} : {}".format(ticker, company.get_uncertainty()))
    
    try:
        fig, ax = plt.subplots(figsize=(14,10)
                              ,dpi=300)
        st.plotly_chart(fig_price)
        
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

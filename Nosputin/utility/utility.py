import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go

from Nosputin.companyDataset import *
from data.extractor import *
from data.ListSnP500 import *

def snp500_filter(values, ticker, performances):
    """
    SnP500 in ListSnP500.py
    """
    is_hit = False
    for i in range(len(ticker)):
        try:
            for t in SnP500:
                if ticker[i].strip() == t.strip():
                    is_hit = True
            if is_hit:
                pass
            else:
                values.pop(i)
                performances.pop(i)
                ticker.pop(i)
            is_hit = False    
        except IndexError:
            continue
    
    st.write("SNP500 filter:: {} companies total".format(len(values)))
    return values, ticker, performances


def xls_to_jsons(filename, t_start="2000-01-01", t_end="2019-12-12"):
    dataset = SimFinDataset(filename, 'comma', t_start, t_end, False)
    timelines = dataset.timePeriods
    company_set = dataset.companies

    for c in company_set:
        pd_c = c.unpack(timelines=timelines).dropna(how='all')
        pd_c.to_json("data/companies/{}.json".format(c.ticker))

    for c in company_set:
        try:
            f = open("data/companies/{}.json".format(c.ticker), encoding="UTF-8")
        except:
            print("Failed to open {}".format(c.ticker))
            continue
        val = json.loads(f.read())
        f.close()

        val['ticker'] = c.ticker

        f = open("data/companies/{}.json".format(c.ticker), 'w', encoding="UTF-8")
        json_val = json.dump(val, f)
        f.close()

        
def show_dist(dataset, sep_date=None):
    #def load_for_dist(data, sep_date):
    #    return get_raw_value(data, sep_date=sep_date)
    #raw_values, tickers, _ = load_for_dist(dataset, sep_date)
    
    raw_values, _, _ = dataset.get_raw_value(sep_date=sep_date)
    
    lengths = []
    length_count = {}
    for v in raw_values:
        lengths.append(len(v))
    for l in set(lengths):
        length_count[l] = lengths.count(l)
    
    fig , ax = plt.subplots(figsize=(12,10), dpi=300)
    ax = plt.bar(length_count.keys(), length_count.values(), align='center', alpha=0.8, width=20)
    plt.style.use('ggplot')
    plt.title("Dimensions of companies", fontsize=30)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel("Number of reported features", fontsize=20)
    plt.ylabel("Number of companies", fontsize=20)
    st.pyplot(dpi=300)

    return lengths


def show_plotly(x, y, text, name="", max_y=100, min_y=0):
    color_max = min(max(y),max_y)
    
    scatter_fig = go.Figure(data=go.Scattergl(x=np.array(x[0]),
                                   y=np.array(x[1]),
                                   mode='markers',
                                   marker=dict(size=8,
                                           color=y,
                                          cmax=color_max,
                                          cmin=min_y,
                                          colorscale='Viridis',
                                          showscale=True),
                                   text=text))
    
    scatter_fig.update_layout(title="Latent space view of {}".format(name),
                     margin=dict(l=0,r=0,b=0),
                     coloraxis=dict(cmin=0,cmax=100))
    st.plotly_chart(scatter_fig)
    dist_fig = ff.create_distplot([y] ,[name])
    dist_fig.update_layout(margin=dict(l=0,r=0,b=0))
    st.plotly_chart(dist_fig)
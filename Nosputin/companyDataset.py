import pandas as pd
import numpy as np
import re
import streamlit as st
import scipy
from datetime import datetime
from math import floor
import math

class NanPerformanceError(Exception):
    def __init__(self, message=None):
        self.message = message

class Company(object):
    def __init__(self, ticker, prices, data, industry):
        self.ticker = ticker
        self.prices = prices
        self.data = data
        self.startDateComplete = None
        self.helperComplete = True
        self.industry = industry

    def __str__(self):
        return "id: " + str(self.id) + ", name: " + str(self.name) + ",  ticker: " + str(self.ticker) + ", data: " + ",".join(str(x) for x in self.data)
    
    def tolist(self, price=False):
        if price:
            return self.prices.values.reshape(-1).tolist()
        else:
            return self.data.values.reshape(-1).tolist()
        
    def sep_data(self, sep_date):
        sep_date = datetime.strptime(sep_date,"%Y-%m-%d")
        sep_a = Company(self.ticker, self.prices.loc[self.prices.index < sep_date], self.data.loc[self.data.index < sep_date], self.industry)
        sep_b = Company(self.ticker, self.prices.loc[self.prices.index >= sep_date], self.data.loc[self.data.index >= sep_date], self.industry)
        return sep_a, sep_b
    
    def get_performance(self, sep_date=None, snp500=False):
        if sep_date != None:
            data = self.sep_data(sep_date)[1].data
        else:
            data = self.data
        prices = self.prices.dropna(how='any')
        start = prices.iat[0]
        end = prices.iat[-1]
        performance = (end - start)/start * 100
        if math.isnan(performance):
            raise NanPerformanceError
        else:
            return (end - start)/start * 100
        
    def select_data_by_date(self, start_date=None, end_date=None):
        if start_date != "":
            start_date = datetime.strptime(start_date,"%Y-%m-%d")
        if end_date != "":
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        #self.data = self.data.loc[start_date : end_date]
        self.data = self.data[(self.data.index >= start_date) & (end_date >= self.data.index)]
        self.prices = self.prices[(self.prices.index >= start_date) & (end_date >= self.prices.index)]

    def to_nparr(self):
        return self.data.dropna(how='any').values.reshape(-1)
    
    def get_uncertainty(self):
        prices = list(self.prices)
        d_prices = np.diff(prices)
        
        diff_percent = []
        for x,y in zip(prices[1:], d_prices):
            diff_percent.append(y/x*100)
        derivation = math.sqrt(np.var(diff_percent))
        if math.isnan(derivation):
            raise NanPerformanceError
        else:
            return derivation 

    
class CompanyDataset:
    def __init__(self):
        self.numIndicators = None
        self.numCompanies = 1
        self.indicators = []
        self.companies = []
        self.tickers = []
        self.timePeriods = []
        self.snp500Price = None
        self.startDatetime = None
        self.sepDatetime = None
        self.endDatetime = None
    
    def loadData_mongo(self, db_client):
        tickers_from_db = db_client.COMPANIES.find({}, {"_id":0, "ticker":1})
        possible_tickers = []
        for t in tickers_from_db:
            if t['ticker'] != "":
                possible_tickers.append(t['ticker'])
				
        indicator_dict = {'_id':0,
             'Share Price':1,
             'Revenues':1,
             'COGS':1,
             'SG&A':1,
             'EBIT':1,
             'EBITDA':1,
             'Net Income from Discontinued Op.':1,
             'Net Profit':1,
             'Dividends':1,
             'Cash and Cash Equivalents':1,
             'Receivables':1,
             'Current Assets':1,
             'Net PP&E':1,
             'Intangible Assets':1,
             'Goodwill':1,
             'Total Noncurrent Assets':1,
             'Total Assets':1,
             'Short term debt':1,
             'Accounts Payable':1,
             'Current Liabilities':1,
             'Long Term Debt':1,
             'Total Noncurrent Liabilities':1,
             'Total Liabilities':1,
             'Preferred Equity':1,
             'Share Capital':1,
             'Treasury Stock':1,
             'Retained Earnings':1,
             'Equity Before Minorities':1,
             'Total Equity':1,
             'Cash From Operating Activities':1,
             'Cash From Investing Activities':1,
             'Cash From Financing Activities':1,
             'Net Change in Cash':1,
             'Industry code': 1}
        
        self.indicators = list(db_client.COMPANIES.find_one({}, indicator_dict))
        self.numIndicators = len(self.indicators) - 2
        
        maximum = db_client.COMPANIES.count()
        st.write("Get the dataset from Nosputin database . . .")
        prog = st.progress(0)
        i = 0
        
        for t in possible_tickers:
            dict_data = db_client.COMPANIES.find_one({"ticker":t},indicator_dict)
            
            industry = str(dict_data['Industry code'])
            dict_data.pop('Industry code')
            pd_price = pd.Series(dict_data['Share Price'], dtype=np.float64).dropna(how='any')
            pd_price.index = pd.to_datetime(pd_price.index)
            
            dict_data.pop('Share Price')
            pd_data = pd.DataFrame.from_dict(dict_data, dtype=np.float64).dropna(how='any')
            if len(pd_data) == 0:
                i += 1
                continue
            pd_data.index = pd.to_datetime(pd_data.index)
            self.companies.append(Company(t, pd_price, pd_data, industry))
            self.tickers.append(t)
            i += 1
            prog.progress(i/maximum)
        
        self.numCompanies = len(self.companies)
        
    def loadSnp500Price(self):
        self.snp500Price = pd.read_csv("./data/SNP500_hist.csv", index_col=0)[['Close']]
        self.snp500Price.index = pd.to_datetime(self.snp500Price.index)
        self.snp500Price = self.snp500Price['Close']
        
    def getCompany(self, ticker):
        for i in range(len(self.companies)):
            if self.tickers[i] == ticker:
                return self.companies[i]
        return None
    
    def select_dimension(self, sel_dim):
        i = 0
        while True:
            if i == len(self.companies) - 1:
                break
                
            if sel_dim is not None and sel_dim !=0:
                if len(self.companies[i].tolist()) != sel_dim:
                    self.companies.pop(i)
                else:
                    i = i + 1
    
    def relative_performance(self, performance, sep_date=None):
        """
        Calculate relative performances.
        """
        if sep_date != None:
            selected_snp500 = self.snp500Price[(self.snp500Price.index >= self.sepDatetime)
                                           & (self.snp500Price.index < self.endDatetime)]
        else:
            selected_snp500 = self.snp500Price[(self.snp500Price.index >= self.startDatetime)
                                           & (self.snp500Price.index < self.endDatetime)]
        start = float(selected_snp500.iat[0])
        end = float(selected_snp500.iat[-1])
        performance_snp500 = (end - start)/start * 100
        
        return performance - performance_snp500
        
    
    def get_raw_value(self, sel_dim=None, sep_date=None, noise_call=False):
        raw_values = []
        tickers = []
        performances = []
        if noise_call:
            noises = []
        if sep_date != None:
            self.sepDatetime = sep_date
        
        st.write("Get raw value of companies . . .")

        prog = st.progress(0)
        i = 0
        for c in self.companies:
            i += 1
            if sel_dim is not None and sel_dim != 0:
                if len(c.tolist()) != sel_dim:
                    continue
            try:
                if sep_date != None:
                    c_a, c_b = c.sep_data(sep_date)
                else:
                    c_a = c
                    c_b = c
                Y = c_b.get_performance(sep_date=sep_date)
                Y = self.relative_performance(Y, sep_date=sep_date)
                if noise_call:
                    N = c_a.get_uncertainty()
                T = c_a.ticker
                X = c_a.tolist()
                raw_values.append(X)
                performances.append(Y)
                tickers.append(T)
                if noise_call:
                    noises.append(N)
            except IndexError:
                continue
            except NanPerformanceError:
                continue
            except Exception as ex:
                raise ex 
            prog.progress(int(i/len(self.companies)*100))

        if noise_call:
            return raw_values, tickers, performances, noises
        return raw_values, tickers, performances
    
    def export_training_data(self):
        pass
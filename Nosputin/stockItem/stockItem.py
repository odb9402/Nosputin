import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class stockItem():

    def __init__(self, ticker, exchangeChart='WIKI', exchangeFund='RAYMOND', collapseSet='weekly'):
        """

        :param ticker:
        :param exchangeChart:
        :param exchangeFund:
        :param collapseSet:
        """
        quandl.ApiConfig.api_key = '9jxfZ6agSLz1orBkN-JV'
        ticker = ticker.upper()
        self.symbol = ticker
        try:
            chartData = quandl.get('{}/{}'.format(exchangeChart, ticker.replace(' ','')), collapse=collapseSet)
        except Exception as e:
            print('Error Retrieving Data')
            print(e)
            raise EnvironmentError

        chartData = chartData.reset_index(level=0)
        self.chartData = chartData.copy()
        self.chartData = self.chartData.drop(['Ex-Dividend', 'Split Ratio'], axis=1)

    def __len__(self):
        return len(self.chartData)

    def extractData(self, startIndex, endIndex, predictSize):
        """
        Extract small fragment from stock data with start and end of index
        and divide it with ratio value to make this can be used to train a model.

        :param startIndex:
        :param endIndex:
        :param predictSize:
        :return: Numpy array of data
        """

        data = self.chartData[startIndex:endIndex].drop(['Date'], axis=1)

        priorData = data[0 : len(data) - predictSize]
        postData = data[len(data) - predictSize + 1 : len(data)]

        return (priorData.values, postData.values)
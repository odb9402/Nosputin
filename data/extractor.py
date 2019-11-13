import csv
from math import floor
from datetime import datetime
import pandas as pd
import numpy as np
import re

class Company(object):
    def __init__(self, compId):
        self.id = int(compId)
        self.name = ""
        self.ticker = ""
        self.industryCode = 0
        self.finYearMonthEnd = 0
        self.data = []
        self.startDateComplete = None
        self.helperComplete = True

    def __str__(self):
        return "id: " + str(self.id) + ", name: " + str(self.name) + ",  ticker: " + str(self.ticker) + ", data: " + ",".join(str(x) for x in self.data)
    
    def appendValue(self, indicatorIndex, value):
        self.data[indicatorIndex].values.append(value)

    def setDateComplete(self,indicatorIndex, dateObj):
        self.data[indicatorIndex].startDateComplete = dateObj
        if dateObj is not None and self.helperComplete:
            if self.startDateComplete is None or dateObj > self.startDateComplete:
                self.startDateComplete = dateObj
        else:
            self.helperComplete = False
            self.startDateComplete = None
            
    def unpack(self, timelines=None):
        unpacked = np.transpose(np.array([x.values for x in self.data]))
        names = [x.name for x in self.data]
        if timelines is not None:
            return pd.DataFrame(unpacked, columns=names, index=timelines, dtype=np.float64)
        else:
            return pd.DataFrame(unpacked, columns=names, dtype=np.float64)
    
    def tolist(self):
        return self.unpack().dropna(how='any').values.reshape(-1,).tolist()
    
    def to_nparr(self):
        return self.unpack().dropna(how='any').values.reshape(-1,)

    
class Indicator:
    def __init__(self, name,indicatorId):
        self.name = name
        self.values = []
        self.indicatorId = indicatorId
        self.startDateComplete = None

    def __str__(self):
        return "{name: " + str(self.name) + ", indicatorId: "+str(self.indicatorId)+", len(values): " + str(len(self.values)) + "}"


class SimFinDataset:
    def __init__(self, dataFilePath, csvDelimiter = "semicolon", startDate = "", endDate = "", excludeMissing = False, companyClass=Company):

        self.numIndicators = None
        self.numCompanies = 1

        self.quarterPattern = re.compile(r'(\w)(\d)-(\d{4})')

        self.companies = []
        self.tickers = []
        self.timePeriods = []
        self.timePeriodsDates = []
        self.timePeriodFormat = None

        self.numDescriptionRows = 7
        self.excludeMissing = excludeMissing

        self.startDatetime = None
        self.endDatetime = None
        self.startIndexLimit = None
        self.endIndexLimit = None
        if startDate != "":
            self.startDatetime = datetime.strptime(startDate,"%Y-%m-%d")
        if endDate != "":
            self.endDatetime = datetime.strptime(endDate, "%Y-%m-%d")

        # load data
        self.loadData(dataFilePath, csvDelimiter, companyClass)

        self.numTimePeriods = len(self.timePeriods)

        # if complete companies only are requested, filter out the ones that have missing data
        if excludeMissing:
            cutDate = self.startDatetime if self.startDatetime is not None else self.timePeriodsDates[0]
            for a in range(self.numCompanies-1,-1,-1):
                if self.companies[a].startDateComplete is None or self.companies[a].startDateComplete > cutDate:
                    self.deleteCompanyAtIndex(a)

    def loadData_mongo(self, db_client):
        tickers_query = db_client.COMPANIES.find({}, {"_id":0, "ticker":1})
        tickers = []
        for t in tickers_query:
            tickers.append(list(t.values())[0])
        tickers = set(tickers)
        
        
    def loadData(self, filePath, delimiter, companyClass=Company):

        def getCompIndex(index,numIndicators):
            return int(floor((index - 1) / float(numIndicators)))

        def getIndicatorIndex(index,numIndicators,compIndex):
            return index - 1 - (numIndicators * compIndex)

        numRow = 0

        delimiterChar = ";" if delimiter == "semicolon" else ","

        csvfile = open(filePath, 'r')
        reader = csv.reader(csvfile, delimiter=delimiterChar, quotechar='"')
        row_count = sum(1 for _ in reader)
        csvfile.seek(0)

        for row in reader:
            numRow += 1
            if numRow > 1 and numRow != row_count and numRow != row_count-1:
                # info rows for company
                if numRow <= 7:
                    # company id row
                    if numRow == 2:
                        rowLen = len(row)
                        idVal = None
                        for index, columnVal in enumerate(row):
                            if index > 0:
                                if idVal is not None and idVal != columnVal:
                                    self.numCompanies += 1
                                    if self.numIndicators is None:
                                        self.numIndicators = index - 1
                                    # add last company
                                    self.companies.append(companyClass(idVal))
                                if index + 1 == rowLen:
                                    if self.numIndicators is None:
                                        self.numIndicators = index
                                    # add last company in file
                                    self.companies.append(companyClass(columnVal))
                                idVal = columnVal
                    if numRow > 2 and self.numIndicators is None:
                        return
                    # company name row
                    if numRow == 3:
                        for a in range(0, self.numCompanies):
                            self.companies[a].name = row[(a * self.numIndicators) + 1]
                    # company ticker row
                    if numRow == 4:
                        for a in range(0, self.numCompanies):
                            self.companies[a].ticker = row[(a * self.numIndicators) + 1]
                            self.tickers.append(self.companies[a].ticker)
                    # company financial year end row
                    if numRow == 5:
                        for a in range(0, self.numCompanies):
                            self.companies[a].finYearMonthEnd = row[(a * self.numIndicators) + 1]
                    # company industry code row
                    if numRow == 6:
                        for a in range(0, self.numCompanies):
                            self.companies[a].industryCode = row[(a * self.numIndicators) + 1]
                    # indicator name row
                    if numRow == 7:
                        for a in range(0, self.numCompanies):
                            for b in range(0, self.numIndicators):
                                self.companies[a].data.append(Indicator(row[(a * self.numIndicators + b) + 1],b))
                else:
                    # actual data
                    inDateRange = False
                    for index, columnVal in enumerate(row):
                        if index == 0:

                            # set time period format
                            if self.timePeriodFormat is None:
                                if self.quarterPattern.match(columnVal):
                                    self.timePeriodFormat = "quarters"
                                else:
                                    self.timePeriodFormat = "dates"

                            currentDate = self.getDateFromStr(columnVal)

                            # check if in date range
                            if (self.startDatetime is None or currentDate >= self.startDatetime) and (self.endDatetime is None or currentDate <= self.endDatetime):
                                inDateRange = True

                            if inDateRange:
                                self.timePeriods.append(columnVal)
                                self.timePeriodsDates.append(currentDate)

                        else:

                            compIndex = getCompIndex(index, self.numIndicators)
                            indicatorIndex = getIndicatorIndex(index, self.numIndicators, compIndex)
                            if columnVal == "" or columnVal is None:
                                appendVal = None
                            else:
                                appendVal = columnVal

                            if inDateRange:
                                self.companies[compIndex].appendValue(indicatorIndex, appendVal)

            elif numRow == row_count-1:
                # the "missing values" row is not used here, since the very last row is a better indicator for completeness of the data
                pass
            #in the last row, the date is saved starting at which the indicator is complete, i.e. has no gaps
            elif numRow == row_count:
                for index, columnVal in enumerate(row):
                    if index > 0:
                        compIndex = getCompIndex(index, self.numIndicators)
                        indicatorIndex = getIndicatorIndex(index, self.numIndicators, compIndex)
                        self.companies[compIndex].setDateComplete(indicatorIndex,self.getDateFromStr(columnVal))


    def deleteCompanyAtIndex(self,index):
        del self.companies[index]
        del self.tickers[index]
        self.numCompanies -= 1

    def getCompany(self, ticker):
        if ticker in self.tickers:
            return self.companies[self.tickers.index(ticker)]
        else:
            return None

    def getDateFromStr(self, dateStr):

        if dateStr == "":
            return None

        if self.timePeriodFormat == "quarters":
            match = self.quarterPattern.match(dateStr)
            currentQuarter = int(match.group(2))
            currentYear = int(match.group(3))
            return datetime(currentYear, (currentQuarter - 1) * 3 + 1, 1)
        else:
            # to datetime obj
            return datetime.strptime(dateStr, '%Y-%m-%d')
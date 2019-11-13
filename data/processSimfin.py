#!/usr/bin/env python

import extractor
import getopt,sys,os
import re
try:
    import xlsxwriter
except ImportError:
    print("Can't import xlsxWritter , use \"pip install XlsxWriter\" or visit http://xlsxwriter.readthedocs.io to get it")
    exit()

def nextCol(max):
    col_num=0
    while col_num < max:
        yield col_num
        col_num +=1

def getRelevantComapnies(dataset, tickers):
    if tickers != None:
        relevant_companies = set()
        for ticker in tickers:
            if ticker not in dataset.tickers:
                print("The ticker ", ticker, " doesn't exist in db")
            else:
                ticker_index = dataset.tickers.index(ticker)
                relevant_companies.add((ticker_index, dataset.companies[ticker_index]))
        return relevant_companies
    
    print ("Using all companies in Dataset - num companies is %u" % (len(dataset.companies)))
    return enumerate(dataset.companies)
    

def parseDb(input_file_name, delmiterStr, minYear, tickers = None):

    print ("Reading Dataset ...")
    dataset = extractor.SimFinDataset(input_file_name,delmiterStr)
    print ("Done Reading Dataset")
    
    file_name = os.path.splitext(input_file_name)[0] # remove extension
    xlsx_file_name = '%s.xlsx' % file_name
    print("Creating %s" % xlsx_file_name)
    workbook = xlsxwriter.Workbook(xlsx_file_name)
    worksheet = workbook.add_worksheet()

    indicator_name_list = []
    
    col_gen = nextCol(100)
    worksheet.write(0, next(col_gen), "Name")
    worksheet.write(0, next(col_gen), "Ticker")
    for indicator in dataset.companies[0].data:
        worksheet.write(0, next(col_gen), indicator.name)
        indicator_name_list.append(indicator.name)

    period_name_pattern = "(\w)(\d)-(\d{4})"
    date_string_pattern = "(\d{4})-(\d{2})-(\d{2})"
    period_name_re = re.compile(period_name_pattern)
    date_string_re = re.compile(date_string_pattern)
        
    periodIdxList = []
    # find relevant periods and list 
    for periodIdx,time_period in enumerate(dataset.timePeriods):
        fin_period_match = period_name_re.match(time_period)
        date_string_match = date_string_re.match(time_period)
        if (fin_period_match):
            year = int(fin_period_match.group(3))
        elif (date_string_match):
            year = int(date_string_match.group(1))
        else:
            year = "NA"
        
        if (year >= minYear): #include this time period
            periodIdxList.append(periodIdx)
            #print("Will append data period %s" % time_period)

    
    print("Done period Idx fetch , num periods found is %d " % len(periodIdxList))
    
    worksheet.write(0, next(col_gen), "Period Name")
    worksheet.write(0, next(col_gen), "Report Year")
    
    numMissingIndicators = 0
    num_columns = next(col_gen) - 1
    
    numCompanies = len(dataset.companies)
    row = 1
    for companyIdx,company in getRelevantComapnies(dataset, tickers):
        for periodIdx in periodIdxList:
            time_period = dataset.timePeriods[periodIdx]
            #print "Writing period %s" % time_period
            col_gen = nextCol(100)
            worksheet.write(row, next(col_gen), company.name)
            worksheet.write(row, next(col_gen), company.ticker)
            for indIdx,indicator in enumerate(company.data):
                if indicator.name != indicator_name_list[indIdx]:
                    print("%s in not in initial list for ticker %s" % (indicator.name,company.ticker))
                    numMissingIndicators += 1
                    worksheet.write(row, next(col_gen), "NA")
                else:
                    if (indicator.values[periodIdx] == None):
                        worksheet.write(row, next(col_gen), "NA")
                    else:
                        worksheet.write(row, next(col_gen), indicator.values[periodIdx])
            worksheet.write(row, next(col_gen), time_period)
            row += 1
        print("Written Fundamnetals for Company %d/%d (%d%%)" % (companyIdx,numCompanies,100*companyIdx/numCompanies))

    if tickers is not None:                    
        print("Num companies written %d , Num data periods %d - num missing indicators %d , num row written %u, collumns %d" % (companyIdx,dataset.numTimePeriods,numMissingIndicators,row,num_columns))
    print("File saved as %s" % xlsx_file_name)
    
    #freeze top row :
    worksheet.freeze_panes(1, 0)
    worksheet.set_column(0,num_columns, 15) # set column width 

    
    # apply autofilter:
    worksheet.autofilter(0,0,row,num_columns)
    workbook.close()


def print_usage():
    print("--inputFile=<> - specify the CSV to be parsed (mandatory)")
    print("--delimiter=<> - specify the CSV delimiter (defaults to semicolon)")
    print("--minYear=<> - will only include entries from this year onwards")
    print("--tickers=<> - will only include entries from this year onwards")
    print("--help - print this information")

def main():
    input_file_name = None
    minYear = 0
    tickers = None
    delimiter = "semicolon"
    try:
        opts, args = getopt.getopt(sys.argv[1:] ,'', ["help","inputFile=","delimiter=","minYear=","tickers=", "ticker="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -a not recognized"
        print_usage()
        sys.exit(2)
    for opt, val in opts:
        if opt == "--help":
            print_usage()
            return
        elif opt in ("--inputFile"):
            input_file_name = val
            print("Will read from %s" % input_file_name)
        elif opt in ("--delimiter"):
            delimiter = val
        elif opt in ("--minYear"):
            minYear = int(val)
            print("Will ignore reports from years before %u" % minYear)
        elif opt in ("--tickers", "-ticker"):
            print("tickers to be extracted are %s" % val)
            tickers = val.split(",")
        else:
            assert False, "unknown option %s" % opt 

    if (input_file_name != None):
        import os.path
        if (os.path.isfile(input_file_name)):
            pass             
        else:
            print("%s doesn't exist" % input_file_name)
            return
        parseDb(input_file_name,delimiter,minYear, tickers)
    else:
        print("No input file name given!")
        print_usage()
        return
    


main()
    

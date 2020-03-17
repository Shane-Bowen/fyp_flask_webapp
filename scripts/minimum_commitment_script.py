#!/usr/bin/env python
#############################################################################################################################
#
# Author: Shane Bowen
#
# Objective: This script will retrieve the minimum monthly commitments for each company and write to csv file
#
# Date: 04/02/2020
# 
#############################################################################################################################

# Import libraries
import os
import re
import sys
import csv
import pandas as pd
import numpy as np
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import MySQLdb
import json
from pathlib import Path

def getCompanyId(company):
    # This function returns the company_id given the name of a company

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, 'config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT id FROM company
                WHERE name = '{0}'""".format(company)
    print(sql)
        
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchone()

    #print(results)
    return results

def min_commit(start_date, end_date):
    # This function formats the company and it's minimum commit by month and writes it to a csv file
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../reports/minimum_commits_report.csv')
    f = open(filename, "w+")
    parent = os.path.abspath('..')
    writer = csv.writer(f)
    #writer.writerow(['Minimum Commits Report'])

    company_ids = {}
    min_commit_dict = {}
    heading = ['company']

    cur_date = start_date
    while cur_date < end_date:
        print(cur_date)
        cur_date += relativedelta(months=1)
        last_day_of_month = cur_date - relativedelta(days=1)
        heading.append(last_day_of_month.strftime("%Y-%m-%d"))
        df = pd.read_csv("""./billing/companies_report_{1}.csv""".format(last_day_of_month), encoding = "ISO-8859-1")
        print(df[['Company', 'Min Commit']])
        for i in range(len(df)) : 
            print(df.iloc[i]['Company'], df.iloc[i]['Min Commit'])
            if df.iloc[i]['Company'] in company_ids:
                company_id = company_ids[df.iloc[i]['Company']]
            else:
                company_id = getCompanyId(df.iloc[i]['Company'])
                company_ids[df.iloc[i]['Company']] =  company_id
            if company_id is not None:
                print(company_id[0])
                if company_id[0] in min_commit_dict: #if key is already in dictionary
                    if np.isnan(float(df.iloc[i]['Min Commit'])):
                        min_commit_dict[company_id[0]][last_day_of_month] = ''
                    else:
                        min_commit_dict[company_id[0]][last_day_of_month] = float(df.iloc[i]['Min Commit'])
                else:
                    if np.isnan(float(df.iloc[i]['Min Commit'])):
                        min_commit_dict[company_id[0]] = {}
                        min_commit_dict[company_id[0]][last_day_of_month] = ''
                    else:
                        min_commit_dict[company_id[0]] = {}
                        min_commit_dict[company_id[0]][last_day_of_month] = float(df.iloc[i]['Min Commit'])

    pprint.pprint(min_commit_dict)
    writer.writerows([heading])
   
    # Write to CSV
    for company in sorted(min_commit_dict.keys()):
        print(company)
        row = [company]
        cur_date = start_date
        while cur_date < end_date:
            print(cur_date)
            cur_date += relativedelta(months=1)
            last_day_of_month = cur_date - relativedelta(days=1)
            print(last_day_of_month)
            if last_day_of_month in min_commit_dict[company]: #If dictionary has key, get value
                row.append(min_commit_dict[company][last_day_of_month])
            else: #If key not present, then set value to blank
                row.append('')
        print(row)
        writer.writerows([row])

if __name__ == "__main__":
    # arguments
    # 1- start_date
    # 2- end_date

    # Check if the number of arguements passed is greater than 2, if not then set start and end date to last month's date
    if len(sys.argv) > 2:
        start_date = datetime.strptime('%s' % sys.argv[1] ,'%Y-%m-%d').date()
        end_date = datetime.strptime('%s' % sys.argv[2] ,'%Y-%m-%d').date() 
    else:
        yesterday = datetime.today() - timedelta(months=1)
        start_date = datetime(yesterday.year,yesterday.month,yesterday.day)
        end_date = datetime(yesterday.year,yesterday.month,yesterday.day)

    # Call the min_commit function with parameters passed in
    min_commit(start_date, end_date)
#!/usr/bin/env python
#############################################################################################################################
#
# Author: Shane Bowen
#
# Objective: This script will generate a CSV for daily company data which will be used in a machine learning model
#
# Date: 07/02/2020
# 
#############################################################################################################################

# Import libraries
import os
import re
import sys
import csv
import time
import json
from datetime import datetime, timedelta
import MySQLdb
import json
import pandas as pd
import numpy as np
import operator
from dateutil.relativedelta import relativedelta
from statistics import mean

def get_config_file():
    # This function gets the config file which stores credentials to connect to db
    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

    return data

def get_company_name(company_id):
    # This function gets the company name
    print("###################################################################################")
    print("Get Company Name")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT name FROM company WHERE id = {0} """.format(company_id)
    print(sql)
        
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchone()

    print(results)
    return results

def getCompanyType(company_id):
    # This function returns what type of company it is
    print("###################################################################################")
    print("Get Company Type")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT ct.name FROM company_extension AS ce 
                LEFT JOIN company_type AS ct ON ct.id = ce.company_type_id
                WHERE ce.company_id = {0} """.format(company_id)
    print(sql)
        
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchone()

    if results is None:
        results = (float('NaN'), )

    print(results)
    return results

def getPesqTable(company_id):
    # This function returns an object conatining only the test type id and corresponding pesq table for a company
    # whos application id is not in Manual Testing or External Quality
    print("###################################################################################")
    print("Get PESQ Table")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT tt.id, tt.pesq_table FROM test_type AS tt 
                LEFT JOIN test_type_for_company AS ttc ON ttc.test_type_id = tt.id
                WHERE ttc.company_id = {0} AND tt.status = 1 AND tt.application_id NOT IN (3,9) AND tt.pesq_table != ""
                GROUP BY tt.pesq_table""".format(company_id)
    print(sql)
        
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    print(results)
    return results

def getPesqScores(pesq_table, company_id, start_date, end_date, values_dict):
    # This function calculates the avg. pesq score per day and appends in to dictionary
    print("###################################################################################")
    print("Find Avg. Pesq Scores")

    #Each pesq table has a job_processing table related to it
    print(pesq_table)
    jp_table = pesq_table.replace("pesq", "job_processing")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT DATE_FORMAT(jp.call_start_time, '%Y-%m-%d') AS callStartTime, avg(pe.PESQ_score) as avgPesqScore
            FROM {0} AS jp
                LEFT JOIN number AS n on n.id = jp.number_id
                LEFT JOIN company AS c ON c.id = n.company_id
                LEFT JOIN {1} AS pe ON pe.{0}_id = jp.id
            WHERE n.company_id = {2} AND
                jp.call_start_time BETWEEN '{3} 00:00:00' AND '{4} 23:59:59' AND
                jp.show = 1 AND
                jp.processing_complete = 1
			GROUP BY DATE_FORMAT(jp.call_start_time, '%Y-%m-%d')
            ORDER BY jp.call_start_time ASC""".format(jp_table, pesq_table, company_id, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    print(results)
    for val in results:
        print(val[0]) # date
        print(val[1]) # avg pesq_score

        if 'pesq' in values_dict[str(val[0])]: #if key is already in dictionary
                if val[1] is not None:
                    print([values_dict[str(val[0])]['pesq'], val[1]])
                    print(mean([values_dict[str(val[0])]['pesq'], val[1]]))
                    values_dict[str(val[0])]['pesq'] = mean([values_dict[str(val[0])]['pesq'], val[1]]) #calculate new avg.
        else:
            if val[1] is not None:
                values_dict[str(val[0])]['pesq'] = val[1] # add to dictionary
    
    return values_dict

def daily_quality_too_poor(jp_table, company_id, start_date, end_date, values_dict):
    #This function calculates the daily number of quality too poor's and stores results in a dictionary
    #A tests fails if it returns busy i.e. call_description_id = 5
    print("###################################################################################")
    print("Daily Number of Quality Too Poor")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d') AS call_start_time,
            count(jp.id) as volume_tests
        FROM
            {0} AS jp
                LEFT JOIN
            number n on n.id = jp.number_id
        WHERE
            n.company_id = {1} AND
            jp.call_start_time BETWEEN '{2} 00:00:00' AND '{3} 23:59:59' AND
            jp.call_description_id = 5 AND
            jp.show = 1 AND
            jp.processing_complete = 1
        GROUP BY
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d')
        ORDER BY
            jp.call_start_time, jp.number_id""".format(jp_table, company_id, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    for val in results:
        print(val[0])
        print(val[1])

        if 'quality_too_poor' in values_dict[str(val[0])]: #if key is already in dictionary
            values_dict[str(val[0])]['quality_too_poor'] = values_dict[str(val[0])]['quality_too_poor'] + val[1]
        else:
            values_dict[str(val[0])]['quality_too_poor'] = val[1]

    return values_dict

def daily_number_busy(jp_table, company_id, start_date, end_date, values_dict):
    #This function calculates the daily number of busy's and stores results in a dictionary
    #A tests fails if it returns busy i.e. call_description_id = 3
    print("###################################################################################")
    print("Daily Number of Busy")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d') AS call_start_time,
            count(jp.id) as volume_tests
        FROM
            {0} AS jp
                LEFT JOIN
            number n on n.id = jp.number_id
        WHERE
            n.company_id = {1} AND
            jp.call_start_time BETWEEN '{2} 00:00:00' AND '{3} 23:59:59' AND
            jp.call_description_id = 3 AND
            jp.show = 1 AND
            jp.processing_complete = 1
        GROUP BY
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d')
        ORDER BY
            jp.call_start_time, jp.number_id""".format(jp_table, company_id, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    for val in results:
        print(val[0])
        print(val[1])

        if 'busy' in values_dict[str(val[0])]: #if key is already in dictionary
            values_dict[str(val[0])]['busy'] = values_dict[str(val[0])]['busy'] + val[1]
        else:
            values_dict[str(val[0])]['busy'] = val[1]

    return values_dict

def daily_number_unable(jp_table, company_id, start_date, end_date, values_dict):
    #This function calculates the daily number of temporarily unable to test and stores results in a dictionary
    #A tests is temporarily unable to test if call_description_id = 3
    print("###################################################################################")
    print("Daily Number of Temporarily unable to test")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d') AS call_start_time,
            count(jp.id) as volume_tests
        FROM
            {0} AS jp
                LEFT JOIN
            number n on n.id = jp.number_id
        WHERE
            n.company_id = {1} AND
            jp.call_start_time BETWEEN '{2} 00:00:00' AND '{3} 23:59:59' AND
            jp.call_description_id = 9 AND
            jp.show = 1 AND
            jp.processing_complete = 1
        GROUP BY
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d')
        ORDER BY
            jp.call_start_time, jp.number_id""".format(jp_table, company_id, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    for val in results:
        print(val[0])
        print(val[1])

        if 'unable' in values_dict[str(val[0])]: #if key is already in dictionary
            values_dict[str(val[0])]['unable'] = values_dict[str(val[0])]['unable'] + val[1]
        else:
            values_dict[str(val[0])]['unable'] = val[1]

    return values_dict
    
def findInitialOutage(jp_table, company_id, start_date, end_date):
    # This function returns an object containing numbers with first time busys in corresponding job processing table
    print("###################################################################################")
    print("Find Initial Outage")

    # set the start_date to 1 month ago
    start_date -= relativedelta(months=1)

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT jp.id jobProcessingID, n.id AS numberID, jp.call_start_time AS callStartTime, '{0}' as jobProcessingTable
            FROM {0} AS jp
                LEFT JOIN number AS n on n.id = jp.number_id
            WHERE n.company_id = {1} AND
                jp.call_start_time BETWEEN '{2} 00:00:00' AND '{3} 23:59:59' AND
                jp.call_description_id = 3 AND
                jp.show = 1 AND
                jp.processing_complete = 1
            GROUP BY jp.number_id
            ORDER BY jp.call_start_time ASC""".format(jp_table, company_id, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()
    
    #print(results)
    return results

def findOutageEnd(fail, outage_threshold, company_id, start_date, end_date):
    # This function searches for the end of the outage by using a shifting window method on tests until success criteria for outage finished has been met
    print("###################################################################################")
    print("Find Outage End")

    # set the start date 1 month ago
    start_date -= relativedelta(months=1)

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    outage_end = False
    initial_fail = fail[0]
    jp_start_point = fail[0]
    fu_start_point = fail[0]

    # Loop until outage end criteria has been met or window for tests is less than threshold
    while outage_end is False:
        fail_count = 0
        sql = """SELECT '{3}' AS initialFail, jp.id jobProcessingID, jp.test_type_id AS testTypeID, n.id AS numberID, jp.call_start_time AS callStartTime, 
                        jp.call_connect_time AS callConnectTime, jp.call_description_id AS callDescriptionID
                    FROM {0} AS jp
                        LEFT JOIN number AS n on n.id = jp.number_id
                    WHERE n.company_id = {5} AND 
                        jp.call_start_time BETWEEN '{7} 00:00:00' AND '{8} 23:59:59' AND
                        jp.show = 1 AND
                        jp.test_type_id != 6 AND
                        jp.processing_complete = 1 AND
                        jp.id > {1} AND
                        jp.number_id = {2}
                UNION
                    SELECT '{3}' AS initialFail, jp.id jobProcessingID, jp.test_type_id AS testTypeID, n.id AS numberID, 
                        jp.call_start_time AS callStartTime, jp.call_connect_time AS callConnectTime, jp.call_description_id AS callDescriptionID
                    FROM job_processing_connection AS jp
                        LEFT JOIN number AS n on n.id = jp.number_id
                    WHERE n.company_id = {5} AND 
                        jp.call_start_time BETWEEN '{7} 00:00:00' AND '{8} 23:59:59' AND
                        jp.test_type_id = 6 AND
                        jp.show = 1 AND
                        jp.processing_complete = 1 AND
                        jp.id > {4} AND
                        jp.number_id = {2}
                ORDER BY callStartTime ASC LIMIT {6}""".format(fail[3], jp_start_point, fail[1], initial_fail, fu_start_point, company_id, outage_threshold, fail[2], end_date)
        print(sql)

        # Execute the SQL command
        cursor.execute(sql)

        # Fetch all the rows in a list of lists.
        window = cursor.fetchall()

        # Check if amount of test in window is less than the outage threshold, if so loop through tests in window and check else assume outage is still occuring so calculate outage until midnight
        if len(window) == int(outage_threshold):
            for test in window:
                if test[6] == 3:
                    fail_count = fail_count +1
            
            # If the fail count is greater than 0 then outage is still present else outage has finished so we calculate outage duration and return that value
            if fail_count > 0:
                if window[0][2] == 6:
                    fu_start_point = window[0][1]
                else:
                    jp_start_point = window[0][1]
                continue
            else:
                print("***Outage has finished***")
                outage_start_time = datetime.strptime(str(fail[2]), '%Y-%m-%d %H:%M:%S')
                outage_end_time = datetime.strptime(str(window[0][4]), '%Y-%m-%d %H:%M:%S')
                outage_duration =  outage_end_time - outage_start_time
                outage_end = True   
                return outage_duration
        else:
            print("***Outage Is Continuing next day***") 
            outage_start_time = datetime.strptime(str(fail[2]), '%Y-%m-%d %H:%M:%S')
            outage_end_time = datetime.strptime(str(end_date) + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
            outage_duration =  outage_end_time - outage_start_time
            outage_end = True
            return outage_duration

def daily_outage(fail, outage_duration, company_id, start_date, end_date, values_dict):
    # This function calculates the daily outages duration and stores it inside a dictionary
    print("###################################################################################")
    print("Find Daily Outages")

    value_dict_key = start_date
    start_date -= relativedelta(months=1)

    day_start_time = fail[2].replace(hour=0,second=0,minute=0)
    outage_start_time = datetime.strptime(str(fail[2]), '%Y-%m-%d %H:%M:%S')
    day_end_time = fail[2].replace(hour=23,second=59,minute=59)

    #keep looping until outage duration is 0
    while outage_duration > timedelta(seconds=0):

        if (outage_start_time + outage_duration) > day_end_time: #if start_time and duration is greater than end_time, then calculate duration just until midnight
            print("Outage Duration too big")
            that_day_duration = day_end_time - outage_start_time
            print(that_day_duration)

            if start_date <= day_start_time.date() <= end_date:
                if day_start_time.date() >= value_dict_key:
                    if 'outage' in values_dict[day_start_time.strftime("%Y-%m-%d")]: #if key is already in dictionary
                        values_dict[day_start_time.strftime("%Y-%m-%d")]['outage'] = values_dict[day_start_time.strftime("%Y-%m-%d")]['outage'] + that_day_duration
                    else:
                        values_dict[day_start_time.strftime("%Y-%m-%d")]['outage'] = that_day_duration

                day_start_time = day_start_time + timedelta(days=1) #add 1 day to time
                outage_start_time = day_start_time
                day_end_time = day_end_time + timedelta(days=1)
                outage_duration = outage_duration - that_day_duration #subtract outage for that day from total outage duration
            else:
                break

        else:
            print("Outage Duration fits")

            if start_date <= day_start_time.date() <= end_date:
                if day_start_time.date() >= value_dict_key:
                    if 'outage' in values_dict[day_start_time.strftime("%Y-%m-%d")]:
                        values_dict[day_start_time.strftime("%Y-%m-%d")]['outage'] = values_dict[day_start_time.strftime("%Y-%m-%d")]['outage'] + outage_duration
                        print(values_dict[day_start_time.strftime("%Y-%m-%d")]['outage'] + outage_duration)
                    else:
                        values_dict[day_start_time.strftime("%Y-%m-%d")]['outage'] = outage_duration
                        print(outage_duration)
                outage_duration = 0
                break
            else:
                break
            
    return values_dict

def daily_numbers_tested(jp_table, company_id, start_date, end_date, values_dict):
    #This function calculates the daily amount of numbers and stores results in dictionary
    print("###################################################################################")
    print("Daily Volume Tests")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d') AS call_start_time,
            count(DISTINCT jp.number_id) as volume_numbers
        FROM
            {0} AS jp
                LEFT JOIN
            number n on n.id = jp.number_id
        WHERE
            n.company_id = {1} AND
            jp.call_start_time BETWEEN '{2} 00:00:00' AND '{3} 23:59:59' AND
            jp.show = 1 AND
            jp.processing_complete = 1
        GROUP BY
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d')
        ORDER BY
            jp.call_start_time, jp.number_id""".format(jp_table, company_id, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    for val in results:
        print(val[0])
        print(val[1])

        if 'numbers' in values_dict[str(val[0])]: #if key is already in dictionary
            values_dict[str(val[0])]['numbers'] = values_dict[str(val[0])]['numbers'] + val[1]
        else:
            values_dict[str(val[0])]['numbers'] = val[1]

    return values_dict

def daily_followup_tests(jp_table, company_id, start_date, end_date, values_dict):
    #This function calculates the daily volume of Follow up tests and stores results in a dictionary
    print("###################################################################################")
    print("Daily Volume Follow Up Tests")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d') AS call_start_time,
            count(jp.id) as followup_tests
        FROM
            {0} AS jp
                LEFT JOIN
            number n on n.id = jp.number_id
        WHERE
            n.company_id = {1} AND
            jp.call_start_time BETWEEN '{2} 00:00:00' AND '{3} 23:59:59' AND
            jp.test_type_id = 6 AND
            jp.show = 1 AND
            jp.processing_complete = 1
        GROUP BY
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d')
        ORDER BY
            jp.call_start_time, jp.number_id""".format(jp_table, company_id, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    for val in results:
        print(val[0])
        print(val[1])

        if 'followup' in values_dict[str(val[0])]: #if key is already in dictionary
            values_dict[str(val[0])]['followup'] = values_dict[str(val[0])]['followup'] + val[1]
        else:
            values_dict[str(val[0])]['followup'] = val[1]

    return values_dict

def getMinCommit(company_id, start_date, end_date):
    #This functions retrieves the min_commit for a company given a time range and return a list of min_commits
    print("###################################################################################")
    print("Get Min Commit")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    
    # read csv
    dirname = os.path.dirname(__file__)
    min_commit_file = os.path.join(dirname, '../reports/minimum_commits_report.csv')
    df = pd.read_csv(min_commit_file) 
    results = df.loc[df['company'] == int(company_id)]

    # if newer date and not in csv file, then read from db
    if end_date > datetime.strptime(results.keys()[-1], '%Y-%m-%d').date():
        sql = """SELECT
            min_commitment
        FROM
            company_billing_with_call_bundle
        WHERE
            company_id = {0} AND
            min_commitment > 0""".format(company_id)

        print(sql)

        # Execute the SQL command
        cursor.execute(sql)

        val = cursor.fetchone() 
        # if val is None, set min_commit to nan
        if val is None:
            min_commit = float('NaN')
        # else set min_commit to first val
        else:
            min_commit = float(val[0])

        # append key and min_commit to result df
        last_date_of_month = datetime(end_date.year, end_date.month, 1) + relativedelta(months=1, days=-1)
        key = last_date_of_month.strftime("%Y-%m-%d") #convert to string
        results[key] = min_commit
    
    # if no min_commit found, set dummy value to ensure dataframe is the proper format
    if results.empty:
        print("true")
        cars = {'company': [company_id],
                '2018-01-31': [float('NaN')]
                }
        results = pd.DataFrame(cars, columns = ['company', '2018-01-31'])

    return results

def getJobProcessingTable(company_id):
    # This function returns an object conatining only the test type id and corresponding job processing table for a company
    # whos application id is not in Manual Testing or External Quality
    print("###################################################################################")
    print("Get Job Processing Table")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT tt.id, tt.job_processing_table FROM test_type AS tt 
                LEFT JOIN test_type_for_company AS ttc ON ttc.test_type_id = tt.id
                WHERE ttc.company_id = {0} AND tt.status = 1 AND tt.application_id NOT IN (3,9) 
                GROUP BY tt.job_processing_table""".format(company_id)
    print(sql)
    
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    #print(results)
    return results

def daily_volume_tests(jp_table, company_id, start_date, end_date, values_dict):
    #This function calculates the daily volume of tests and stores results in a dictionary
    print("###################################################################################")
    print("Daily Volume Tests")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d') AS call_start_time,
            count(jp.id) as volume_tests
        FROM
            {0} AS jp
                LEFT JOIN
            number n on n.id = jp.number_id
        WHERE
            n.company_id = {1} AND
            jp.call_start_time BETWEEN '{2} 00:00:00' AND '{3} 23:59:59' AND
            jp.show = 1 AND
            jp.processing_complete = 1
        GROUP BY
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d')
        ORDER BY
            jp.call_start_time, jp.number_id""".format(jp_table, company_id, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()
    
    for val in results:
        print(val[0]) #call_start_time
        print(val[1]) #volume_tests

        if 'volume' in values_dict[str(val[0])]: #if key is already in dictionary
            values_dict[str(val[0])]['volume'] = values_dict[str(val[0])]['volume'] + val[1]
        else:
            values_dict[str(val[0])]['volume'] = val[1]
        
        if 'test_types' in values_dict[str(val[0])]:
            if jp_table not in values_dict[str(val[0])]['test_types']:
                print(jp_table)
                print(values_dict[str(val[0])]['test_types'])
                values_dict[str(val[0])]['test_types'].append(jp_table)
        else:
            values_dict[str(val[0])]['test_types'] = [jp_table]
                
    return values_dict

def daily_manual_tests(company_id, start_date, end_date, values_dict):
    #This function gets information for daily manual tests and stores result in a dictionary
    print("###################################################################################")
    print("Daily Manual Tests")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT
            DATE_FORMAT(jp.call_start_time, '%Y-%m-%d') as call_start_time,
            count(jp.id) AS volume_tests,
            sum(if(jp.call_end_reason=0,1,0)) AS temp_unable_test,
            sum(if(jp.call_end_reason=3,1,0)) AS number_busy,
            sum(if(jp.call_end_reason=1,1,0)) AS user_cancelled
	    FROM
            job_processing_manual AS jp
	        LEFT JOIN user AS u ON u.id = jp.user_id
	        LEFT JOIN company AS c ON c.id = u.company_id
	        LEFT JOIN test_type AS tt ON tt.id = jp.test_type_id
	    WHERE
            jp.call_start_time >= '{1} 00:00:00' and jp.call_start_time <= '{2} 23:59:59' AND
			jp.show = 1 AND
			u.company_id = {0}
	    GROUP BY DATE_FORMAT(jp.call_start_time, '%Y-%m-%d')""".format(company_id, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    for val in results:
        print(val[0]) #date
        print(val[1]) #volume_tests
        print(val[2]) #temp_unable_tests
        print(val[3]) #number_busy

        if 'volume' in values_dict[str(val[0])]:
            values_dict[str(val[0])]['volume'] = values_dict[str(val[0])]['volume'] + val[1]
        else:
            values_dict[str(val[0])]['volume'] = val[1]
        if 'unable' in values_dict[str(val[0])]:
            values_dict[str(val[0])]['unable'] = values_dict[str(val[0])]['unable'] + val[2]
        else:
            values_dict[str(val[0])]['unable'] = val[2]
        if 'busy' in values_dict[str(val[0])]:
            values_dict[str(val[0])]['busy'] = values_dict[str(val[0])]['busy'] + val[3]
        else:
            values_dict[str(val[0])]['busy'] = val[3]

    return values_dict

def get_company_ids():
    #This function returns all active company ids for processing
    print("###################################################################################")
    print("Get Company Ids")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    
    # SQL statement
    sql = """SELECT id FROM company WHERE status = 1"""
        
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    # append company_ids to array
    company_ids = []
    for val in results:
        company_ids.append(val[0])

    print(company_ids)
    return company_ids

def ignore_company_ids():
    #This function returns all the company ids to ignore for processing
    print("###################################################################################")
    print("Ignore Company Ids")

    # get config file for db
    data = get_config_file()

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    
    # SQL statement
    sql = """SELECT company_id FROM manual_online_company"""
        
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    # append company_ids to array
    company_ids = []
    for val in results:
        company_ids.append(val[0])
    company_ids.extend([8,36,151,223,281])

    print(company_ids)
    return company_ids

# # sort csv with multiple customer data by time
def sort_csv():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../reports/company_report.csv")
    reader = csv.reader(open(filename), delimiter=",")
    header = next(reader)
    sortedlist = sorted(reader, key=operator.itemgetter(3))
    print(len(sortedlist))

    filename = os.path.join(dirname, "../reports/company_report_sorted.csv")
    writer = csv.writer(open(filename, "w+"))
    writer.writerow(header)
    for row in sortedlist:
        print(row)
        writer.writerow(row)

def main(company_list, outage_threshold, start_date, end_date, ignore_companies):
    # This function calls all the different functions, stores result in a dictionary and writes to a csv file in the current directory (company_report.csv)  

    not_list = []
    # for val in range(1, 9):
    #     not_list.append(val)

    # loop through each company_id in list
    for company_id in company_list:
        if company_id not in ignore_companies:
            if company_id not in not_list:
                dirname = os.path.dirname(__file__)
                filename = os.path.join(dirname, "../reports/company_report.csv")
                
                # check if file exists (append to existing file)
                if os.path.isfile(filename):
                    f = open(filename, "a+")
                    writer = csv.writer(f)
                else: # create new file
                    f = open(filename, "w+")
                    writer = csv.writer(f)
                    writer.writerow(['company_name', 'company_id', 'company_type', 'time', 'date', 'month', 'year', 'day', 'is_weekend', 'season', 'avg_pesq_score', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'followup_tests', 'min_commit', 'has_min_commit', 'volume_tests', 'is_testing'])

                # Get company_name and company_type
                company_name = get_company_name(company_id)[0]
                company_type = getCompanyType(company_id)[0]

                # Intitialzie Dictionary
                values_dict = {}
                delta = timedelta(days=1)
                cur_date = start_date
                while cur_date <= end_date:
                    values_dict[str(cur_date)] = {}
                    cur_date += delta
                
                # Get Pesq Scores
                for table in getPesqTable(company_id):
                    print(table)
                    values_dict = getPesqScores(table[1], company_id, start_date, end_date, values_dict)
                print(values_dict)

                # Populate values dict with input features
                for table in getJobProcessingTable(company_id):
                    values_dict = daily_volume_tests(table[1], company_id, start_date, end_date, values_dict)
                    values_dict = daily_quality_too_poor(table[1], company_id, start_date, end_date, values_dict)
                    values_dict = daily_number_busy(table[1], company_id, start_date, end_date, values_dict)
                    values_dict = daily_number_unable(table[1], company_id, start_date, end_date, values_dict)
                    values_dict = daily_numbers_tested(table[1], company_id, start_date, end_date, values_dict)
                    values_dict = daily_followup_tests(table[1], company_id, start_date, end_date, values_dict)
                    values_dict = daily_manual_tests(company_id, start_date, end_date, values_dict)
                    for fail in findInitialOutage(table[1], company_id, start_date, end_date):
                        print(fail)
                        outage_duration = findOutageEnd(fail, outage_threshold, company_id, start_date, end_date)
                        values_dict = daily_outage(fail, outage_duration, company_id, start_date, end_date, values_dict)

                # Get company_min_commit
                company_min_commit = getMinCommit(company_id, start_date, end_date)
                print(values_dict)

                # Set boolean variable
                ok = False
                
                # Write to CSV File
                for date, dictionary in values_dict.items():

                    # don't write empty rows to csv file (i.e. wait until first record for volume of tests)
                    if 'volume' in dictionary:
                        ok = True
                    
                    # if record found, start writing to csv file
                    if ok:
                        cur_date = datetime.strptime(date, '%Y-%m-%d') #convert to date object
                        last_date_of_month = datetime(cur_date.year, cur_date.month, 1) + relativedelta(months=1, days=-1)
                        key = last_date_of_month.strftime("%Y-%m-%d") #convert to string

                        # get data information
                        time = date
                        date = cur_date.day
                        month = cur_date.month
                        year = cur_date.year

                        # define months for seasons
                        spring = [2, 3, 4]
                        summer = [5, 6, 7]
                        autumn = [8, 9, 10]
                        winter = [11, 12, 1]

                        # define weekdays
                        weekDays = ("Mon","Tues","Wed","Thurs","Fri","Sat","Sun")
                        weekday = [0, 1, 2, 3, 4]

                        # check if day is weekend
                        day = weekDays[cur_date.weekday()]
                        if cur_date.weekday() in weekday:
                            isWeekend = 0
                        else:
                            isWeekend = 1

                        # check what season month is
                        season = ''
                        if month in spring:
                            season = 'Spring'
                        elif month in summer:
                            season = 'Summer'
                        elif month in autumn:
                            season = 'Autumn'
                        elif month in winter:
                            season = 'Winter'

                        # boolean value to check if performed any tests, default is 1
                        is_testing = 1

                        # if key not in dictionary, set value to 0
                        if 'pesq' not in dictionary: 
                            dictionary['pesq'] = 0
                        if 'quality_too_poor' not in dictionary:
                            dictionary['quality_too_poor'] = 0
                        if 'busy' not in dictionary:
                            dictionary['busy'] = 0
                        if 'unable' not in dictionary:
                            dictionary['unable'] = 0
                        if 'outage' not in dictionary:
                            dictionary['outage'] = 0
                        if 'test_types' not in dictionary:
                            dictionary['test_types'] = []
                        if 'numbers' not in dictionary:
                            dictionary['numbers'] = 0
                        if 'followup' not in dictionary:
                            dictionary['followup'] = 0
                        if 'volume' not in dictionary:
                            dictionary['volume'] = 0
                            is_testing = 0

                        # check if has min_commit
                        has_min_commit = 1
                        if key not in company_min_commit:
                            company_min_commit[key] = float('NaN')
                            has_min_commit = 0

                        if np.isnan(company_min_commit[key].values[0]) or company_min_commit[key].values[0] == 0:
                            has_min_commit = 0

                        # convert days to hours
                        if('outage' in dictionary and dictionary['outage'] != 0):
                            hours = round(float(dictionary['outage'].total_seconds() / 3600), 2) # convert to hours
                        else:
                            hours = 0
                        
                        # write to csv
                        writer.writerow([company_name, company_id, company_type, time, date, month, year, day, isWeekend, season, dictionary['pesq'], dictionary['quality_too_poor'], dictionary['busy'], dictionary['unable'], hours, len(dictionary['test_types']), dictionary['numbers'], dictionary['followup'], company_min_commit[key].values[0], has_min_commit, dictionary['volume'], is_testing])
    sort_csv()
    print("Script Finished")
    f.close()

if __name__ == "__main__":
    # arguments
    # 1- company
    # 2- outage_threshold
    # 3- start_date
    # 4- end_date
    
    # outage threshold & start_date args
    if len(sys.argv) == 3:
        company_list = get_company_ids()
        outage_threshold = sys.argv[1]
        start_date = datetime.strptime('%s' % sys.argv[2] ,'%Y-%m-%d').date()
        end_date = start_date

    # outage threshold, start date and end date args
    elif len(sys.argv) == 4:
        company_list = get_company_ids()
        outage_threshold = sys.argv[1]
        start_date = datetime.strptime('%s' % sys.argv[2] ,'%Y-%m-%d').date()
        end_date = datetime.strptime('%s' % sys.argv[3] ,'%Y-%m-%d').date()

    # company list, outage threshold, start date and end date args
    elif len(sys.argv) == 5:
        if '[' in sys.argv[1]:
            company_list = list(sys.argv[1].strip('[]').split(","))
        else:
            company_list = list(sys.argv[1])
        outage_threshold = sys.argv[2]
        start_date = datetime.strptime('%s' % sys.argv[3] ,'%Y-%m-%d').date()
        end_date = datetime.strptime('%s' % sys.argv[4] ,'%Y-%m-%d').date()

    # company list to ignore
    ignore_companies = ignore_company_ids()
        
    # Call the outage function with parameters passed in
    main(company_list, outage_threshold, start_date, end_date, ignore_companies)

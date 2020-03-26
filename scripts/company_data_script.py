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

def getJobProcessingTable(company):
    # This function returns an object conatining only the test type id and corresponding job processing table for a company
    # whos application id is not in Manual Testing or External Quality

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, 'config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    print("###################################################################################")
    print("Get Job Processing Table")

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT tt.id, tt.job_processing_table FROM test_type AS tt 
                LEFT JOIN test_type_for_company AS ttc ON ttc.test_type_id = tt.id
                WHERE ttc.company_id = {0} AND tt.status = 1 AND tt.application_id NOT IN (3,9) 
                GROUP BY tt.job_processing_table""".format(company)
    print(sql)
    
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    #print(results)
    return results

def getPesqTable(company):
    # This function returns an object conatining only the test type id and corresponding pesq table for a company
    # whos application id is not in Manual Testing or External Quality

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    print("###################################################################################")
    print("Get PESQ Table")

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT tt.id, tt.pesq_table FROM test_type AS tt 
                LEFT JOIN test_type_for_company AS ttc ON ttc.test_type_id = tt.id
                WHERE ttc.company_id = {0} AND tt.status = 1 AND tt.application_id NOT IN (3,9) AND tt.pesq_table != ""
                GROUP BY tt.pesq_table""".format(company)
    print(sql)
        
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()

    print(results)
    return results

def getCompanyType(company):
    # This function returns what type of company it is

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    print("###################################################################################")
    print("Get Company Type")

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """SELECT ct.name FROM company_extension AS ce 
                LEFT JOIN company_type AS ct ON ct.id = ce.company_type_id
                WHERE ce.company_id = {0} """.format(company)
    print(sql)
        
    # Execute the SQL command    
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchone()

    print(results)
    return results

def getPesqScores(pesq_table, company, start_date, end_date, values_dict):
    # This function calculates the avg. pesq score per day and appends in to dictionary
    
    print("###################################################################################")
    print("Find Avg. Pesq Scores")

    #Each pesq table has a job_processing table related to it
    print(pesq_table)
    jp_table = pesq_table.replace("pesq", "job_processing")

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

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
            ORDER BY jp.call_start_time ASC""".format(jp_table, pesq_table, company, start_date, end_date)
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

def daily_quality_too_poor(jp_table, company, start_date, end_date, values_dict):
    #This function calculates the daily number of quality too poor's and stores results in a dictionary
    #A tests fails if it returns busy i.e. call_description_id = 5

    print("###################################################################################")
    print("Daily Number of Quality Too Poor")

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

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
            jp.call_start_time, jp.number_id""".format(jp_table, company, start_date, end_date)
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

def daily_number_busy(jp_table, company, start_date, end_date, values_dict):
    #This function calculates the daily number of busy's and stores results in a dictionary
    #A tests fails if it returns busy i.e. call_description_id = 3

    print("###################################################################################")
    print("Daily Number of Busy")

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

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
            jp.call_start_time, jp.number_id""".format(jp_table, company, start_date, end_date)
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

def daily_number_unable(jp_table, company, start_date, end_date, values_dict):
    #This function calculates the daily number of temporarily unable to test and stores results in a dictionary
    #A tests is temporarily unable to test if call_description_id = 3

    print("###################################################################################")
    print("Daily Number of Temporarily unable to test")

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

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
            jp.call_start_time, jp.number_id""".format(jp_table, company, start_date, end_date)
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
    
def findInitialOutage(jp_table, company, start_date, end_date):
    # This function returns an object containing numbers with first time busys in corresponding job processing table

    print("###################################################################################")
    print("Find Initial Outage")

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

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
            ORDER BY jp.call_start_time ASC""".format(jp_table, company, start_date, end_date)
    print(sql)

    # Execute the SQL command
    cursor.execute(sql)

    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()
    
    #print(results)
    return results

def findOutageEnd(fail, outage_threshold, company, start_date, end_date):
    # This function searches for the end of the outage by using a shifting window method on tests until success criteria for outage finished has been met

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

    # Open database connection
    db = MySQLdb.connect(data['mysql']['host'], data['mysql']['user'], data['mysql']['passwd'], data['mysql']['db'])

    print("###################################################################################")
    print("Find Outage End")

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
                ORDER BY callStartTime ASC LIMIT {6}""".format(fail[3], jp_start_point, fail[1], initial_fail, fu_start_point, company, outage_threshold, fail[2], end_date)
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

def daily_outage(fail, outage_duration, company, start_date, end_date, values_dict):
    # This function calculates the daily outages duration and stores it inside a dictionary

    print("###################################################################################")
    print("Find Daily Outages")

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

def daily_numbers_tested(jp_table, company, start_date, end_date, values_dict):
    #This function calculates the daily amount of numbers and stores results in dictionary

    print("###################################################################################")
    print("Daily Volume Tests")

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

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
            jp.call_start_time, jp.number_id""".format(jp_table, company, start_date, end_date)
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

def daily_followup_tests(jp_table, company, start_date, end_date, values_dict):
    #This function calculates the daily volume of Follow up tests and stores results in a dictionary

    print("###################################################################################")
    print("Daily Volume Follow Up Tests")

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

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
            jp.call_start_time, jp.number_id""".format(jp_table, company, start_date, end_date)
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

def getMinCommit(company, start_date, end_date):
    #This functions retrieves the min_commit for a company given a time range and return a list of min_commits
    
    dirname = os.path.dirname(__file__)
    min_commit_file = os.path.join(dirname, '../billing/minimum_commits_report.csv')
    df = pd.read_csv(min_commit_file)
    results = df.loc[df['company'] == int(company)]

    print(results)
    return results

def daily_manual_tests(company, start_date, end_date, values_dict):
    #This function gets information for daily manual tests and stores result in a dictionary

    print("###################################################################################")
    print("Daily Manual Tests")

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

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
	    GROUP BY DATE_FORMAT(jp.call_start_time, '%Y-%m-%d')""".format(company, start_date, end_date)
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

def daily_volume_tests(jp_table, company, start_date, end_date, values_dict):
    #This function calculates the daily volume of tests and stores results in a dictionary

    print("###################################################################################")
    print("Daily Volume Tests")

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
    with open(config_file) as json_data_file:
        data = json.load(json_data_file)

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
            jp.call_start_time, jp.number_id""".format(jp_table, company, start_date, end_date)
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

# # sort csv with multiple customer data by time
# def sort_csv():
#     reader = csv.reader(open("../reports/company_report.csv"), delimiter=",")
#     header = next(reader)
#     sortedlist = sorted(reader, key=operator.itemgetter(3))
#     print(len(sortedlist))

#     writer = csv.writer(open("../reports/company_report_sorted.csv", "w+"))
#     writer.writerow(header)
#     for row in sortedlist:z
#         print(row)
#         writer.writerow(row)

def main(company_list, outage_threshold, start_date, end_date):
    # This function calls all the different functions, stores result in a dictionary and writes to a csv file in the current directory (company_report.csv)  

    for company in company_list:

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, """../reports/company_report_{}.csv""".format(company))
        #filename = os.path.join(dirname, """../reports/company_report.csv""")
        print(filename)
        f = open(filename, "w+")
        writer = csv.writer(f)
        writer.writerow(['volume_tests', 'company_id', 'company_type', 'time', 'date', 'month', 'year', 'day', 'is_weekend', 'season', 'avg_pesq_score', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'followup_tests', 'min_commit'])


        company_min_commit = getMinCommit(company, start_date, end_date)
        company_type = getCompanyType(company)[0]

        #Intitialzie Dictionary
        values_dict = {}
        delta = timedelta(days=1)
        cur_date = start_date
        while cur_date <= end_date:
            values_dict[str(cur_date)] = {}
            cur_date += delta
        
        #Get Pesq Scores
        for table in getPesqTable(company):
            print(table)
            values_dict = getPesqScores(table[1], company, start_date, end_date, values_dict)
        print(values_dict)

        #Get Outage Duration
        for table in getJobProcessingTable(company):
            values_dict = daily_quality_too_poor(table[1], company, start_date, end_date, values_dict)
            values_dict = daily_number_busy(table[1], company, start_date, end_date, values_dict)
            values_dict = daily_number_unable(table[1], company, start_date, end_date, values_dict)
            values_dict = daily_numbers_tested(table[1], company, start_date, end_date, values_dict)
            values_dict = daily_followup_tests(table[1], company, start_date, end_date, values_dict)
            values_dict = daily_manual_tests(company, start_date, end_date, values_dict)
            values_dict = daily_volume_tests(table[1], company, start_date, end_date, values_dict)
            for fail in findInitialOutage(table[1], company, start_date, end_date):
                print(fail)
                outage_duration = findOutageEnd(fail, outage_threshold, company, start_date, end_date)
                values_dict = daily_outage(fail, outage_duration, company, start_date, end_date, values_dict)

        #Write to CSV File
        for date, dictionary in values_dict.items():
            cur_date = datetime.strptime(date, '%Y-%m-%d') #convert to date object
            last_date_of_month = datetime(cur_date.year, cur_date.month, 1) + relativedelta(months=1, days=-1)
            key = last_date_of_month.strftime("%Y-%m-%d") #convert to string

            time = date
            date = cur_date.day
            month = cur_date.month
            year = cur_date.year

            spring = [2, 3, 4]
            summer = [5, 6, 7]
            autumn = [8, 9, 10]
            winter = [10, 11, 1]

            weekDays = ("Mon","Tues","Wed","Thurs","Fri","Sat","Sun")
            weekday = [0, 1, 2, 3, 4]

            day = weekDays[cur_date.weekday()]
            if cur_date.weekday() in weekday:
                isWeekend = 0
            else:
                isWeekend = 1

            if month in spring:
                season = 'Spring'
            elif month in summer:
                season = 'Summer'
            elif month in autumn:
                season = 'Autumn'
            elif month in winter:
                season = 'Winter'

            if 'pesq' not in dictionary: #if key not in dictionary
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

            #convert days to hours
            if('outage' in dictionary and dictionary['outage'] != 0):
                hours = round(float(dictionary['outage'].total_seconds() / 3600), 2) # convert to hours
            else:
                hours = 0
            
            writer.writerow([dictionary['volume'], company, company_type, time, date, month, year, day, isWeekend, season, dictionary['pesq'], dictionary['quality_too_poor'], dictionary['busy'], dictionary['unable'], hours, len(dictionary['test_types']), dictionary['numbers'], dictionary['followup'], company_min_commit[key].values[0]])
    #sort_csv()
    print("Script Finished")
    f.close()

if __name__ == "__main__":
    # arguments
    # 1- company
    # 2- outage_threshold
    # 3- start_date
    # 4- end_date

    if '[' in sys.argv[1]:
        company_list = list(sys.argv[1].strip('[]').split(","))
    else:
        company_list = list(sys.argv[1])

    outage_threshold = sys.argv[2]

    # Check if the number of arguements passed is greater than 4, if not then set start and end date to last month's date
    if len(sys.argv) > 2:
        start_date = datetime.strptime('%s' % sys.argv[3] ,'%Y-%m-%d').date()
        end_date = datetime.strptime('%s' % sys.argv[4] ,'%Y-%m-%d').date()
    else:
        yesterday = datetime.today() - timedelta(months=1)
        start_date = datetime(yesterday.year,yesterday.month,yesterday.day)
        end_date = datetime(yesterday.year,yesterday.month,yesterday.day)
    
    # Call the outage function with parameters passed in
    main(company_list, outage_threshold, start_date, end_date)
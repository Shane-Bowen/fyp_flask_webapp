#!/usr/bin/env python
#############################################################################################################################
#
# Author: Shane Bowen
#
# Objective: This script will calculate the daily outage duration and volume of tests for a customer
#
# Date: 28/01/2020
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

def getJobProcessingTable(company):
     # This function returns an object conatining only the test type id and corresponding job processing table for a company
    # whos application id is not in Manual Testing or External Quality

    dirname = os.path.dirname(__file__)
    config_file = os.path.join(dirname, './config/config.json')
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

def outage(company, outage_threshold, start_date, end_date):
    # This functions finds the duration of the outage and writes to a csv file in the current directory (outage_report.csv) and sends the report  

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, """../reports/outage_report_{}.csv""".format(company))
    f = open(filename, "w+")
    writer = csv.writer(f)
    writer.writerow(['Outage Report'])
    writer.writerow(['date', 'company_id', 'outage_hrs'])

    #Intitialzie Dictionary
    outage_dict = {}
    delta = timedelta(days=1)
    cur_date = start_date
    while cur_date <= end_date:
        outage_dict[str(cur_date)] = {}
        cur_date += delta
        
    for table in getJobProcessingTable(company):
        for fail in findInitialOutage(table[1], company, start_date, end_date):
            print(fail)
            outage_duration = findOutageEnd(fail, outage_threshold, company, start_date, end_date)
            outage_dict = daily_outage(fail, outage_duration, company, start_date, end_date, outage_dict)

    for date, dictionary in outage_dict.items():
        if 'outage' not in dictionary:
            dictionary['outage'] = 0

        #convert days to hours
        if('outage' in dictionary and dictionary['outage'] != 0):
            hours = round(float(dictionary['outage'].total_seconds() / 3600), 2) # convert to hours
            minutes = round(float(dictionary['outage'].total_seconds() / 60), 2) # convert to minutes
            seconds = float(dictionary['outage'].total_seconds()) # seconds
        else:
            hours = 0
            minutes = 0
            seconds = 0

        writer.writerow([date, company, hours, minutes, seconds])
    print("Script Finished")
    f.close()

if __name__ == "__main__":
    # arguments
    # 1- company
    # 2- outage_threshold
    # 3- start_date
    # 4- end_date

    # Check if the number of arguements passed is greater than 4, if not then set start and end date to last month's date
    if len(sys.argv) > 4:
        start_date = datetime.strptime('%s' % sys.argv[3] ,'%Y-%m-%d').date()
        end_date = datetime.strptime('%s' % sys.argv[4] ,'%Y-%m-%d').date() 
    else:
        yesterday = datetime.today() - timedelta(months=1)
        start_date = datetime(yesterday.year,yesterday.month,yesterday.day)
        end_date = datetime(yesterday.year,yesterday.month,yesterday.day)

    company = sys.argv[1]          
    outage_threshold = sys.argv[2]
    
    # Call the outage function with parameters passed in
    outage(company, outage_threshold, start_date, end_date)
"""
Process the movebank output files 
"""

import numpy as np
import pandas as pd 
import os 

######## DB settings #################
from pandas.io import sql
import MySQLdb

HOST = '127.0.0.1'
USER = 'root'
PWD = '123456'
DB = 'lab_data'
PORT = 3306

GPS_TABLE = "storks_movebank_gps"
ACC_TABLE = "storks_movebank_acc"
######################################



def process_acc_file(f_name):
    # load the rew data 
    f = pd.DataFrame.from_csv(f_name, index_col=None) 

    # keep only these columns 
    mapping = {'eobs:accelerations-raw':'acc_data', 
               'tag-local-identifier':'tag_id', 
               'individual-local-identifier':'bird_id'}
    cols_keep = list(mapping.keys())
    out = f[cols_keep].copy()
    out.columns = [mapping[key] for key in out.columns]

    # process 
    out['time'] = pd.DatetimeIndex(f['eobs:start-timestamp']).time
    out['date'] = pd.DatetimeIndex(f['eobs:start-timestamp']).date
    out['bird_id'] = out['bird_id'].apply(lambda s: s[:4] + '0' + s[-4:-1])
    out['num_samples'] = out['acc_data'].apply(lambda s: s.count(' ') + 1)
    out['acc_data'] = out['acc_data'].apply(lambda s: s.replace(' ', ','))

    return out

def process_gps_file(f_name):
     # load the rew data 
    f = pd.DataFrame.from_csv(f_name, index_col=None) 

    # keep only these columns 
    mapping = {
        #"event-id": " ",
        #"visible": " ",
        #"timestamp": " ",
        "location-long": "gps_long",
        "location-lat": "gps_lat",
        #"eobs:activity": " ",
        #"eobs:activity-samples": " ",
        "eobs:battery-voltage": "bat_voltage",
        #"eobs:fix-battery-voltage": " ",
        "eobs:horizontal-accuracy-estimate": "horizontal_inacc",
        #"eobs:key-bin-checksum": " ",
        "eobs:speed-accuracy-estimate": "speed_inacc",
        #"eobs:start-timestamp": " ",
        "eobs:status": "gps_status",
        "eobs:temperature": "temperature",
        "eobs:type-of-fix": "type_fix",
        "eobs:used-time-to-get-fix": "time_2_fix",
        "ground-speed": "speed",
        "heading": "heading",
        "height-above-ellipsoid": "hight_elipse",
        #"manually-marked-outlier": " ",
        #"sensor-type": " ",
        #"individual-taxon-canonical-name": " ",
        "tag-local-identifier": "tag_id",
        "individual-local-identifier": "bird_id",
        #"study-name": " ",
        #"utm-easting": " ",
        #"utm-northing": " ",
        #"utm-zone": " ",
        #"study-timezone": " ",
        #"study-local-timestamp": " ",
    }

    cols_keep = list(mapping.keys())
    out = f[cols_keep].copy()
    out.columns = [mapping[key] for key in out.columns]

    # process
    out["date_start_fix"] = pd.DatetimeIndex(f["eobs:start-timestamp"]).date
    out["time_start_fix"] = pd.DatetimeIndex(f["eobs:start-timestamp"]).time
    out["date_end_fix"] = pd.DatetimeIndex(f["timestamp"]).date
    out["time_end_fix"] = pd.DatetimeIndex(f["timestamp"]).time
    out['bird_id'] = out['bird_id'].apply(lambda s: s[:4] + '0' + s[-4:-1])

    return out 

def insert(frame, table):
    # connect to DB
    db = MySQLdb.connect(host = HOST,
                         user = USER,
                         passwd = PWD,
                         db = DB,
                         port = PORT)

    # insert table 
    frame = frame.where(pd.notnull(frame), None) #N/A => None so SQL is happy 
    frame.to_sql(table, db, flavor='mysql', if_exists='append', index=False, chunksize=100 )
   

if __name__ == "__main__":
    gps_file = "gps.csv"
    acc_file = "acc.csv"
    gps_data = process_gps_file(gps_file)
    acc_data = process_acc_file(acc_file)
    insert(gps_data, GPS_TABLE)
    insert(acc_data, ACC_TABLE)
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors: Yun and Ben
Date: 09-10 March
Scope: Reads in original dataset (per min) and creates 2 curated datasets (per hour), because of break.
"""

import time
START_TIME = time.time()

FILE_PATH_IN = 'data/bitstamp_usd.csv' # a ',' separated file with tile line
    # col0: timestamp, col2: high, col3: low, col6: volume_currency, col7: weighted_price;
FILE_PATH_OUT_1 = 'data/curated_dataset_1.csv' # a ',' separated file with title line
    # col0: hour, col1: max.high, col2: min.low, col3: sum.volume, col4: mean.price
    # (all in USD), contains part before break
FILE_PATH_OUT_2 = 'data/curated_dataset_2.csv' # a ',' separated file with title line
    # col0: hour, col1: max.high, col2: min.low, col3: sum.volume, col4: mean.price
    # (all in USD), contains part after break
COUNTER = 60 # collect data from 60 rows (= 1h)

# data for break (time points missing) (found out after running check_if_complete())
DELTA_HOURS = 6473 # see lines 34-35 for explanation
ROW_BREAK = 1585521 # see lines 34-35 for explanation

def check_if_complete(file):
    """
    Checks if there is no time stamp missing at a 60s resolution.
    :param file: file object of the dataset with the Bitcoin data
    :return: list with rows that have a non +60s follower (exception: last row)
    """
    file_list = file.readlines()
    row_positions_list, timestamps_list = [], []
    last_time_stamp = int(file_list[1].split(',')[0])
    for a in range(2, len(file_list)):
        current_time_stamp = int(file_list[a].split(',')[0])
        if not last_time_stamp + COUNTER == current_time_stamp:
            row_positions_list.append(str(a - 1)) # time stamp is missing
            timestamps_list.append(str(last_time_stamp))
        last_time_stamp = current_time_stamp
    return row_positions_list, timestamps_list
    # Between row 1585521 and 1585522 (original dataset, so timestamps
    # 1420449120 and 1420837500) are 269.7d (0.74a, 6473h) missing

try:
    # execute the check on missing data in the original dataset
    with open(FILE_PATH_IN) as fi:
        output_1, output_2 = check_if_complete(fi)
        if len(output_1) == 0:
            print ('Read datafile is complete with respect to the time stamps.')
        else:
            print ('Warning: Read datafile is NOT complete with respect to the time stamps.')
            print ('These positions are not followed by +60s time stamp: ' + ','.join(output_1))
            print('This means, these timestamps are not followed by +60s time stamp: ' + ','.join(output_2))
            print ("Don't worry - correction of this problem (1 break) was included in this script.")
except FileNotFoundError:
    print('File ' + FILE_PATH_IN + ' not existent. I assume you named it differently (1).')

try:
    # open the entire data set and create a new file that will contain our data per hour
    with open(FILE_PATH_IN, 'r') as fi:
        with open(FILE_PATH_OUT_1, 'w') as fo:
            fo.write('hour,max.high,min.low,sum.volume,mean.price\n') # write title line
            fi_list = fi.readlines() # skip title line

            # treat data before row break first
            hour_counter = 0
            for a in range(1, ROW_BREAK + 1):
                row_list = fi_list[a].strip().split(',')
                if (a-1) % COUNTER == 0:
                    if not (a-1) == 0: # prevents that empty line is written for a = 0
                        fo.write('%i,%.2f,%.2f,%.2f,%.2f\n' %(hour_counter, max(high_list), min(low_list), \
                                                         sum(volume_list), sum(price_list) / len(price_list)))
                    high_list, low_list, volume_list, price_list = [float(row_list[2])], [float(row_list[3])], \
                            [float(row_list[6])], [float(row_list[7])] # initiates new lists
                    hour_counter += 1
                else: # fills up lists with data from rows
                    high_list.append(float(row_list[2]))
                    low_list.append(float(row_list[3]))
                    volume_list.append(float(row_list[6]))
                    price_list.append(float(row_list[7]))

            hour_counter += DELTA_HOURS
            # new output dataset

        with open(FILE_PATH_OUT_2, 'w') as fo:
            fo.write('hour,max.high,min.low,sum.volume,mean.price\n') # write title line
            # treat data after row break here
            for a in range(ROW_BREAK + 1, len(fi_list)):
                row_list = fi_list[a].strip().split(',')
                if (a - (ROW_BREAK + 1)) % COUNTER == 0:
                    if not (a - (ROW_BREAK + 1)) == 0:  # prevents that empty line is written for a = ROW_BREAK + 1
                        fo.write('%i,%.2f,%.2f,%.2f,%.2f\n' %(hour_counter, max(high_list), min(low_list), \
                                                         sum(volume_list), sum(price_list) / len(price_list)))
                    high_list, low_list, volume_list, price_list = [float(row_list[2])], [float(row_list[3])], \
                            [float(row_list[6])], [float(row_list[7])] # initiates new lists
                    hour_counter += 1
                else: # fills up lists with data from rows
                    high_list.append(float(row_list[2]))
                    low_list.append(float(row_list[3]))
                    volume_list.append(float(row_list[6]))
                    price_list.append(float(row_list[7]))
except FileNotFoundError:
    print('File ' + FILE_PATH_IN + ' not existent. I assume you named it differently (2).')

print ('Dataset was read in from: ' + FILE_PATH_IN)
print ('Curated dataset (part I) was written to: ' + FILE_PATH_OUT_1)
print ('Curated dataset (part II) was written to: ' + FILE_PATH_OUT_2)
print ('Script was successfully executed in %.2s s.' % (time.time() - START_TIME))

# end of script
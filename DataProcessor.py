#!/usr/bin/env python


# open the entire data set and create a new file that will contain our data per hour
with open("bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv", "r") as D:
    with open("NewData.txt", "w") as new_D:
        # only select data per hour instead of per minute
        minutes = D.readlines()
        hours = minutes[1::60]

        # sum the volume traded

        # the required data is the timestamp, high, low, volume in BTC and value (weighted price)
        for line in hours:
            items = line.split(",")
            new_line = items[0] + "," + items[2] + "," + items[3] + "," + items[5] + "," + items[7]
            new_D.write(str(new_line))

#! /usr/bin/env python 3

#######################################################
#
#
# Gantt_tutorial.py
#
# Tutorial for generating gantt charts using matplotlib
#   Based on tutorial found at:
#   https://towardsdatascience.com/gantt-charts-with-pythons-matplotlib-395b7af72d72
# 
#
# Gerald Eaglin, ULL, 9/2/2021
#
#######################################################


# import required packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# make sure data spreadsheet is in the same directory as this script
# cd into the directory

FILE_PATH = Path.cwd() # define path variable to current working directory

df = pd.read_excel(f'{FILE_PATH}/Gantt_data.xlsx') # open excel file containing gantt data
print(df)

# Define a few more variables to make plotting easier

proj_start = df.Start.min() # project start date

# Places new columns in the data
df['start_num'] = (df.Start-proj_start).dt.days # number of days from project start to task start
df['end_num'] = (df.End-proj_start).dt.days # number of days from project start to end of tasks
df['days_start_to_end'] = df.end_num - df.start_num # days between start and end of each task

print(df)
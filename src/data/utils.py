'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/5/17 
    Python Version: 3.6
'''
import os
import pandas as pd
import numpy as np


# read seed queries
def read_query_from_file(seed_path):
    seed_word_list = []
    with open(seed_path, 'r') as fi:
        line = fi.readline()
        while line:
            key_query = line.strip()
            seed_word_list.append(key_query)
#             print(key_query)
            line = fi.readline()
    seed_word_list.sort()
    return seed_word_list


# merge data files
# add datetime index for input dataframe
def add_datetime_index(input_df, date_format):
    """

    :param input_df: pd.DataFrame
    :return: pd.DataFrame
    """

    input_df.rename(columns={input_df.columns[0]: "date"}, inplace=True)
    input_df.index = pd.to_datetime(input_df['date'], format = date_format)
    input_df.drop(['date'], axis=1, inplace=True)
    return input_df


# read raw data from path
def read_raw_data(raw_file_path, add_datetime=False, norm_col=False, date_format=False):
    trend_data = pd.read_csv(raw_file_path)
    if add_datetime:
        trend_data = add_datetime_index(trend_data, date_format)
    if norm_col:
        pass
        # trend_data = normalize_column(trend_data)
    return trend_data


# select_days from  input temporal spatial data
def select_days(input_df, start_date, end_date):
    import datetime as dt
    """

    :param input_df: pd.DataFrame
    :param single_year: int
    :return: list(int)
    """
    start = input_df.index.searchsorted(dt.date(start_date[0], start_date[1], start_date[2]))
    end = input_df.index.searchsorted(dt.date(end_date[0], end_date[1], end_date[2]))

    selected_days = [k for k in range(start, end)]
    output_df = input_df.iloc[selected_days, :]

    return output_df


# get location output path
def get_location_output_path(template_file_path, state_name):
    file_name = '_' + os.path.basename(template_file_path)
    output_path = os.path.join(os.path.dirname(template_file_path), state_name + file_name)
    return output_path


# get outer join common columns
def outer_concatenate(x_train_all, train_data):
    x_train_all = pd.concat([x_train_all, train_data], join='outer', ignore_index=True, sort=False)
    return x_train_all


# code to process data to select common seed words list
def process_data(input_data, seed_word_list, placeholder_df = None):
    y = input_data.iloc[:, 0]
    if placeholder_df is None:
        trend_fea = input_data[seed_word_list]
    else:
        trend_fea = outer_concatenate(placeholder_df, input_data)
        trend_fea = trend_fea[seed_word_list]
        trend_fea.index = input_data.index
        trend_fea.fillna(np.nan, inplace=True)
    # print information
    # print(trend_fea.shape)
    return y, trend_fea


# create folder if not exist
def create_folder_exist(file_save_folder):
    if not os.path.exists(file_save_folder):
        os.makedirs(file_save_folder)








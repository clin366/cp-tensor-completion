# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import configparser
from configparser import ExtendedInterpolation
import ast
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from src.data.utils import read_query_from_file, get_location_output_path,\
    process_data, select_days, read_raw_data
import pickle


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(config_path, input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('predicting the missing values and save the results')

    # read configuration from config file
    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)

    with open(input_filepath, 'rb') as fi:
        imputed_tensor_array = pickle.load(fi)

    # start and end dates
    train_start_date = [int(k) for k in pars['train_test_split']['train_start_date'].split('-')]
    test_start_date = [int(k) for k in pars['train_test_split']['test_start_date'].split('-')]
    # seed word list
    seed_path = pars['global']['seed_query_path']
    seed_word_list = read_query_from_file(seed_path)
    # create placeholder df for this seed queries rank
    placeholder_df = pd.DataFrame(columns=seed_word_list)

    # read location list from pars
    location_list = ast.literal_eval(pars['global']['locations'])

    # input file
    input_file_folder = pars['global']['search_data_path']

    state_count = 0
    for location_name in location_list:
        location_train_data_path = get_location_output_path(input_file_folder, location_name)
        train_data = read_raw_data(location_train_data_path, add_datetime=True, norm_col=False, date_format='%Y-%m-%d')
        y, trend_fea = process_data(train_data, seed_word_list, placeholder_df=placeholder_df)
        trend_fea = select_days(trend_fea, train_start_date, test_start_date)

        # state save path
        location_save_data_path = get_location_output_path(output_filepath, location_name)
        # fill nas
        trend_fea_array = trend_fea.fillna(np.nan)
        trend_fea_array = np.array(trend_fea_array).T
        na_pos = np.where(np.isnan(trend_fea_array))

        # fill na
        city_imputed_array = imputed_tensor_array[state_count]

        trend_fea_array[na_pos] = city_imputed_array[na_pos]
        city_imputed_pd = pd.DataFrame(columns=trend_fea.columns, data=trend_fea_array.T)
        city_imputed_pd.index = trend_fea.index
        city_imputed_pd.to_csv(location_save_data_path, header=True, index=True)
        state_count += 1


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()







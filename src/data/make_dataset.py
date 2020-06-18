# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import configparser
from configparser import ExtendedInterpolation
import ast
import pandas as pd
import sys
sys.path.append('.')
from src.data.utils import *
import pickle


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(config_path, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read configuration from config file
    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)
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

    # use list to store na and sparsity stats
    na_rate_list = []
    sparsity_list = []

    # use list to save train file
    train_x_list = []
    train_y_list = []

    # loop to get all locations metrics
    for location_name in location_list:
        location_train_data_path = get_location_output_path(input_file_folder, location_name)
        train_data = read_raw_data(location_train_data_path, add_datetime=True, norm_col=False, date_format='%Y-%m-%d')
        y, trend_fea = process_data(train_data, seed_word_list, placeholder_df=placeholder_df)
        trend_fea = select_days(trend_fea, train_start_date, test_start_date).T
        # fill all the nas with np.nan
        trend_fea.fillna(np.nan, inplace=True)
        # convert the dataFrame to numpy array
        trend_array = np.array(trend_fea)

        # count sparsity stats and save the stats to file
        sparsity_pos = np.where(trend_array == 0)
        print(location_name)
        print("NA Ratio Count:")
        location_na_rate = np.isnan(trend_array).sum()/trend_array.size * 100.0
        location_sparsity = sparsity_pos[0].size / trend_array.size * 100.0

        na_rate_list.append(location_na_rate)
        sparsity_list.append(location_sparsity)

        print(str(location_na_rate) + "%")

        train_x_list.append(trend_array)
        train_y_list.append(y)

    # save statistics to reports/stats
    save_report_path = 'reports/stats/na_rate_stat_for_expand_query.csv'
    # create report directory to save report
    create_folder_exist(os.path.dirname(save_report_path))
    # save statistics
    na_stat_pd = pd.DataFrame({'NA_rate': na_rate_list, 'Sparsity': sparsity_list})
    na_stat_pd.index = location_list

    # create tensor array for all train_x
    tensor_array = np.array(train_x_list)
    print("The Array Shape of Total Sensor")
    print(tensor_array.shape)
    with open(output_filepath, 'wb') as fo:
        pickle.dump(tensor_array, fo)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()

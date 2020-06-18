# CP completion model for missing value imputation
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
from src.models.utils import choose_test_index, CP_ALS
import pickle
import time
# all the functions for CP completion


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(config_path, input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('training model from raw data')

    print("111")

    # read configuration from config file
    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)

    # read raw input tensor
    with open(input_filepath, 'rb') as fi:
        location_tensor_array = pickle.load(fi)
    sparse_tensor = np.nan_to_num(location_tensor_array, copy=True, nan=0)
    # create test sparse_tensor
    pos = np.where(sparse_tensor != 0)

    test_index = choose_test_index(pos)
    test_index = list(test_index.astype(int))
    test_pos_tuple = tuple([pos[0][test_index], pos[1][test_index], pos[2][test_index]])
    test_values = sparse_tensor[test_pos_tuple]
    sparse_tensor[test_pos_tuple] = 0
    print(len(np.where(sparse_tensor != 0)[0]))
    print(len(test_pos_tuple[0]))
    rank_info_dict = {}

    for rank in [5, 10, 20, 40, 80]:
        start = time.time()
        print('Rank: ' + str(rank))
        print()
        maxiter = 700
        np.random.seed(10)
        tensor_hat, U, V, X, min_test_cls, min_test_cls_iteration = CP_ALS(sparse_tensor, rank, maxiter, test_info = (test_pos_tuple, test_values))
        end = time.time()
        print('Testing TCS: {:.6}'.format(min_test_cls))
        print('Testing iter: {}'.format(min_test_cls_iteration))
        print('Running time: %d seconds'%(end - start))
        rank_info_dict[rank] = [min_test_cls, min_test_cls_iteration]

    # get best rank and iter
    rank_info_pd = pd.DataFrame(rank_info_dict).T
    min_row = rank_info_pd[rank_info_pd[0] == rank_info_pd[0].min()]
    best_rank = min_row.index[0]
    best_iter = int(min_row[1] + 1)
    print("======== Rank and Iteration ======")
    print("Best Rank: " + str(best_rank))
    print("Best Iter: " + str(best_iter))
    print()

    loc_tensor = np.nan_to_num(location_tensor_array, copy=True, nan=0)
    # create test sparse_tensor
    pos = np.where(loc_tensor != 0)
    print(len(pos[0]))

    np.random.seed(10)
    tensor_hat, U, V, X, min_test_cls, min_test_cls_iteration = CP_ALS(loc_tensor, best_rank, best_iter, test_info=None)

    # save output file
    with open(output_filepath, 'wb') as fo:
        pickle.dump(tensor_hat, fo)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/6/16 
    Python Version: 3.6
'''
import numpy as np
from numpy.linalg import inv as inv

# functions for train_model,py
def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)


def cp_combine(U, V, X):
    return np.einsum('is, js, ts -> ijt', U, V, X)


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')


# ALS algorithm for CP completion
def CP_ALS(sparse_tensor_input, rank, maxiter, test_info=None):
    sparse_tensor = sparse_tensor_input.copy()
    dim1, dim2, dim3 = sparse_tensor.shape
    dim = np.array([dim1, dim2, dim3])
    U = 0.1 * np.random.rand(dim1, rank)
    V = 0.1 * np.random.rand(dim2, rank)
    X = 0.1 * np.random.rand(dim3, rank)

    pos = np.where(sparse_tensor != 0)
    binary_tensor = np.zeros((dim1, dim2, dim3))
    binary_tensor[pos] = 1
    tensor_hat = np.zeros((dim1, dim2, dim3))

    min_test_cls = 999
    min_test_cls_iteration = -1

    for iters in range(maxiter):
        for order in range(dim.shape[0]):
            if order == 0:
                var1 = kr_prod(X, V).T
            elif order == 1:
                var1 = kr_prod(X, U).T
            else:
                var1 = kr_prod(V, U).T
            var2 = kr_prod(var1, var1)
            var3 = np.matmul(var2, ten2mat(binary_tensor, order).T).reshape([rank, rank, dim[order]])
            var4 = np.matmul(var1, ten2mat(sparse_tensor, order).T)
            for i in range(dim[order]):
                var_Lambda = var3[:, :, i]
                inv_var_Lambda = inv((var_Lambda + var_Lambda.T) / 2 + 10e-12 * np.eye(rank))
                vec = np.matmul(inv_var_Lambda, var4[:, i])
                if order == 0:
                    U[i, :] = vec.copy()
                elif order == 1:
                    V[i, :] = vec.copy()
                else:
                    X[i, :] = vec.copy()

        tensor_hat = cp_combine(U, V, X)

        if (iters + 1) % 10 == 0:
            mape = np.sum(np.abs(sparse_tensor[pos] - tensor_hat[pos]) / np.abs(sparse_tensor[pos])) / \
                   sparse_tensor[pos].shape[0]
            mape = np.sum(np.abs(sparse_tensor[pos] - tensor_hat[pos]) / np.abs(sparse_tensor[pos])) / \
                   sparse_tensor[pos].shape[0]
            rmse = np.sqrt(np.sum((sparse_tensor[pos] - tensor_hat[pos]) ** 2) / sparse_tensor[pos].shape[0])
            print('Iter: {}'.format(iters + 1))
            print('Training MAPE: {:.6}'.format(mape))
            print('Training RMSE: {:.6}'.format(rmse))
            print()

            if test_info is not None:
                # print test mape and rmse
                test_pos_tuple, test_values = test_info

                norm_tcs = np.linalg.norm(test_values)
                error_tcs = np.linalg.norm(tensor_hat[test_pos_tuple] - test_values)
                test_tcs = error_tcs / norm_tcs

                test_rmse = np.sqrt(np.sum((test_values - tensor_hat[test_pos_tuple]) ** 2) \
                                    / test_values.shape[0])
                print('Testing TCS: {:.6}'.format(test_tcs))
                print('Testing RMSE: {:.6}'.format(test_rmse))
                print()

                ## stop iteration if smallest test_tcs get

                if test_tcs < min_test_cls:
                    min_test_cls = test_tcs
                    min_test_cls_iteration = iters

                else:
                    if ((iters - min_test_cls_iteration) > 30):
                        break
    return tensor_hat, U, V, X, min_test_cls, min_test_cls_iteration


def choose_test_index_for_location(location_array, test_ratio=0.2):
    np.random.seed(2020)
    location_choice = np.random.choice(location_array, size=int(len(location_array)*test_ratio), replace=False)
    return location_choice


def choose_test_index(pos, num_of_locations=50):
    test_index = np.array([])
    for loc_num in range(num_of_locations):
        location_array = np.where(pos[0] == loc_num)[0]
        location_choice = choose_test_index_for_location(location_array)
        location_choice.sort()
        test_index = np.concatenate((test_index, location_choice), axis=0)
    return test_index





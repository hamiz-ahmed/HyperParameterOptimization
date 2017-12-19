"""
Copyright 2017, University of Freiburg.
Muhammad Hamiz Ahmed <hamizahmed93@gmail.com>

This is the code for Deep Learning Lab Exercise 5
"""



import pickle
import numpy as np
import copy
import random as rand
import matplotlib.pyplot as plt
from robo.fmin import bayesian_optimization


rf = pickle.load(open("./rf_surrogate_cnn.pkl", "rb"))
cost_rf = pickle.load(open("./rf_cost_surrogate_cnn.pkl", "rb"))


def objective_function(x, epoch=40):
    """
        Function wrapper to approximate the validation error of the hyperparameter configurations x by the prediction of a surrogate regression model,
        which was trained on the validation error of randomly sampled hyperparameter configurations.
        The original surrogate predicts the validation error after a given epoch. Since all hyperparameter configurations were trained for a total amount of 
        40 epochs, we will query the performance after epoch 40.
    """
    
    # Normalize all hyperparameter to be in [0, 1]
    x_norm = copy.deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)
    

    x_norm = np.append(x_norm, epoch)
    y = rf.predict(x_norm[None, :])[0]

    return y

def runtime(x, epoch=40):
    """
        Function wrapper to approximate the runtime of the hyperparameter configurations x.
    """
    
    # Normalize all hyperparameter to be in [0, 1]
    x_norm = copy.deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)
    

    x_norm = np.append(x_norm, epoch)
    y = cost_rf.predict(x_norm[None, :])[0]

    return y


def optimize_random_serch(bounds):

    D = len(bounds)  # number of decision variables
    best_output = 9999.0  # initialize the "best found" - both the function value and the x values
    best_runtime = 999999.0
    # best_x = [None] * D

    incumbents_performance = np.zeros((10, 50))
    incumbents_runtime = np.zeros((10, 50))

    for k in range(10):



        for i in range(50):
            new_configuration = [rand.randint(bounds[d][0], bounds[d][1]) for d in range(D)]
            new_output = objective_function(new_configuration)

            if new_output < best_output: # see if it's an improvement
                best_output = new_output

            new_runtime = runtime(new_configuration)
            if new_runtime < best_runtime:
                best_runtime = new_runtime

            incumbents_performance[k][i] = best_output
            incumbents_runtime[k][i] = best_runtime

    mean_performance = np.mean(incumbents_performance, axis=0)
    mean_runtimes = np.mean(incumbents_runtime, axis=0)

    return mean_performance, mean_runtimes


def run_random_search():
    learning_rate_bound = [-6, 0]
    batch_size_bound = [32, 512]
    filter_layer_bound = [4, 10]

    bounds = [learning_rate_bound, batch_size_bound, filter_layer_bound, filter_layer_bound, filter_layer_bound]

    performance, runtimes = optimize_random_serch(bounds)

    return performance, runtimes


def run_bayesian_optimization():
    lower = np.array([-6, 32, 4, 4, 4])
    upper = np.array([0, 512, 10, 10, 10])
    incumbents_performance = np.zeros((10, 50))
    incumbents_runtime = np.zeros((10, 50))

    for i in range(10):
        result = bayesian_optimization(objective_function, lower, upper, num_iterations=50)
        incumbents_performance[i] = result.get('incumbent_values')
        incumbents_runtime[i] = result.get('runtime')

        mean_performance = np.mean(incumbents_performance, axis=0)
        mean_runtime = np.mean(incumbents_runtime, axis=0)

    return mean_performance, mean_runtime


if __name__ == '__main__':
    performance_list_random, runtimes_random = run_random_search()
    performance_list_bayesian, runtimes_bayesian = run_bayesian_optimization()

    #performance plot
    iterations = range(50)
    plt.plot(iterations, performance_list_random, label='random search')
    plt.plot(iterations, performance_list_bayesian, label='bayesian search')
    plt.xlabel('steps')
    plt.ylabel('error')
    plt.legend(loc='lower right')
    plt.show()

    runtimes_random = np.cumsum(runtimes_random)
    runtimes_bayesian = np.cumsum(runtimes_bayesian)

    # runtime plot
    plt.plot(iterations, runtimes_random, label='random search')
    plt.plot(iterations, runtimes_bayesian, label='bayesian search')
    plt.xlabel('steps')
    plt.ylabel('runtimes')
    plt.legend(loc='lower right')
    plt.show()
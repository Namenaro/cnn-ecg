import numpy as np
import scipy
from scipy import spatial
import matplotlib
import matplotlib.pyplot as plt
import math
import sklearn
import json
import codecs, json


def get_m_sigma(patient_id_in_dataset):
    raw_dataset_path = "data_1078.json"  # файл с датасетом
    f = open(raw_dataset_path, 'r')
    data = json.load(f)
    v6Data = data[patient_id_in_dataset]["Leads"]["v6"]["Signal"]
    v6Data= np.array(v6Data)
    return v6Data.mean(), v6Data.std()


def kernel1(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def normalise(y, mu, std):
    y_new = (y - y.mean())/y.std()
    y_new = y_new*std
    y_new = y_new+mu
    print(y_new.mean())
    print(y_new.std())
    return y_new

def get_trajectories(mu_desired, std_desired, num_of_trajectories, nb_of_samples=5000):
    # Independent variable samples
    X = np.expand_dims(np.linspace(-140, 140, nb_of_samples), 1) #<<-----------
    Σ = kernel1(X, X)  # Kernel of data points

    # Assume a mean of 0 for simplicity
    ys = np.random.multivariate_normal(
        mean=np.zeros(nb_of_samples), cov=Σ,
        size=num_of_trajectories)
    ynew = []
    for yi in ys:
        y_new_i = normalise(yi, mu_desired, std_desired)
        ynew.append(y_new_i)
    print ("one more done!")
    return np.array(ynew)

def plot_them(trajs):
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title("Trajectories with mu=" + str(trajs[0].mean()) + ", std= " + str(trajs[0].std()))
    for yi in trajs:
        plt.plot(yi)
        print(yi.mean())
        print(yi.std())
    plt.show()

###############################################
#YS = get_trajectories(mu_desired = -5.475, std_desired=188.577, num_of_trajectories = 5, nb_of_samples=5000)
#plot_them(YS)


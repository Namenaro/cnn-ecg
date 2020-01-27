from numpy import number

import stochastic_proc as sp


import numpy as np
import scipy
from scipy import spatial
import matplotlib
import matplotlib.pyplot as plt
import math
import sklearn
import json
import codecs, json

def save_numpy(file_path, nparray_to_save):
    """
    serialise numpy tensor to file json
    """
    b = nparray_to_save.tolist()
    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,
              indent=4)

def restore_numpy(file_path):
    """
        deserialise numpy tensor from file json
        """
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    a_new = np.array(b_new)
    return a_new


def save_trajectpries(patient_id_in_dataset, num_of_trajectories):
    """
    берет номер пациента ,находит его в датасете, и сохраняет в джсон файл несколько траекторий процесса, с там же ми и сигма как у пациента
    :param patient_id_in_dataset:
    :param num_of_trajectories:
    :return:
    """
    mu_desired, sigma_desired = sp.get_m_sigma(patient_id_in_dataset)
    print("mu, sigma = " + str(mu_desired) + ", " + str(sigma_desired))
    YS = sp.get_trajectories(mu_desired=mu_desired, std_desired=sigma_desired, num_of_trajectories=num_of_trajectories, nb_of_samples=5000)
    file_path = str(patient_id_in_dataset)+"noise.json"  ## your path variable
    save_numpy(file_path, YS)


def get_patients(num_patients):
    """
    полуает желаемое количество пациентов из датасета
    :param num_patients:
    :return: возвращает их айдишники (один массив) и их сигналы (второй массив)
    """
    raw_dataset_path = "data_1078.json"
    f = open(raw_dataset_path, 'r')
    data = json.load(f)
    ids = list(data.keys())
    v6Data = []
    for i in range(num_patients):
        my_signal =data[ids[i]]["Leads"]["v6"]["Signal"]
        if len(my_signal)!=5000:
            continue
        v6Data.append(my_signal)

    v6Data = np.array(v6Data)
    return np.array(ids[0:num_patients]), v6Data

def get_signals_for_patients(patients_signals):
    """

    :param patients_signals: numpy array with v6 signals of some patients
    :return: corresponding unperiodic trajectories
    """
    unperiodics = []
    num_of_them = patients_signals.shape[0]
    for i in range(num_of_them):
        ecg =np.array(patients_signals[i])
        mu_desired = ecg.mean()
        std_desired = ecg.std()
        trajectories = sp.get_trajectories(mu_desired, std_desired, num_of_trajectories=1, nb_of_samples=5000)
        unperiodics.append(trajectories[0])
    return np.array(unperiodics)

def plot_several_signals(several_signals, figname):
    num_of_them = several_signals.shape[0]
    fig, ax = plt.subplots(num_of_them, sharex=True, sharey=True)
    for i in range(num_of_them):
        my_signal = list(several_signals[i])
        ax[i].plot(my_signal)
    plt.savefig(figname)


def get_all_data_for_experiment(num_patients):
    ids, v6Data = get_patients(num_patients)
    unperiodic_signals = get_signals_for_patients(v6Data)
    return ids, v6Data, unperiodic_signals

def save_experimental_data(ecg, unperiodic, ids):
    save_numpy("ecgs.json", ecg)
    save_numpy("unperiodic.json", unperiodic)
    save_numpy("ids.json", ids)
    plot_several_signals(unperiodic, "unperiodic.png")
    plot_several_signals(ecg, "ecg.png")

def restore_experimental_data():
    ecgs = restore_numpy("ecgs.json")
    unperiodic = restore_numpy("unperiodic.json")
    ids = restore_numpy("ids.json")
    return ecgs, unperiodic, ids

if __name__ == "__main__":
    # 1) generate and save all experimental data
    ids, v6Data, unperiodic_signals = get_all_data_for_experiment(num_patients=35)
    save_experimental_data(ecg=v6Data, unperiodic=unperiodic_signals, ids=ids)

    # 2) load previously generated data from file
    ecgs, unperiodic, ids = restore_experimental_data()

    # 3) generate and save seveeal unperiodic trajs for one patient
    #save_trajectpries(patient='50483780', num_of_trajectories=5)



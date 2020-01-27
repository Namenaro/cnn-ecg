import json
import keras
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import random
import keras.backend as K
import tensorflow as tf
import math as mt
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, BatchNormalization,Flatten, Dropout, Lambda, Add, add, AveragePooling1D, Reshape
from keras.models import Model
import BaselineWanderRemoval as bwr
import pickle

dataset_path="data_2033\\data_2033.json" #файл с датасетом
def Openf(link):
    # function for opening some json files
    with open (link,'r') as f:
        data = json.load(f)
        return data


def dell_some_ecg(data, arr):
    for i in arr:
        del data[i]

def healthy(diagnos):
    is_heathy =True
    axis_ok = False
    rythm_ok = False
    for key in diagnos.keys():
        if key == 'electric_axis_normal':
            if diagnos[key] == True:
                axis_ok = True
                continue
        if key == 'regular_normosystole':
            if diagnos[key] == True:
                rythm_ok = True
                continue
        if diagnos[key] == True:
            is_heathy = False
            break
    return axis_ok and rythm_ok and is_heathy


def fix_line(entry):
    FREQUENCY_OF_DATASET = 500
    return bwr.fix_baseline_wander(entry, FREQUENCY_OF_DATASET)

def norm_ecg(signal, mean = 0, sigma = 1):
    signal = np.array(signal)
    return (signal - mean)/sigma    

def modify_dataset_on_healthy(data , mean =0 , sigma=1):
    lead_name = 'i'
    wrong_array = []
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        diag = data[case_id]['StructuredDiagnosisDoc']
        if healthy(diag):
            new_entry = leads[lead_name]['Signal']
            data[case_id]['Leads'][lead_name]['Signal'] = norm_ecg( fix_line(new_entry), mean, sigma)  # modify
        else:
            # если пациент не зворов, то удаляем его из датасета
            wrong_array.append(case_id)
    return wrong_array

def modify_dataset_on_sick(data, mean=0, sigma=1):
    lead_name = 'i'
    wrong_array = []
    for case_id in data.keys():
        diag = data[case_id]['StructuredDiagnosisDoc']
        if healthy(diag): # если здоров, то надо удалить
            wrong_array.append(case_id)
        else:
            # выровнить изолинию
            new_entry = data[case_id]['Leads'][lead_name]["Signal"]
            data[case_id]['Leads'][lead_name]['Signal'] = norm_ecg( fix_line(new_entry), mean, sigma ) # modify
    return wrong_array
    
def modify_dataset(data , mean =0, sigma=1):
    lead_name = 'i'
    wrong_array = []
    for case_id in data.keys():
        new_entry = data[case_id]['Leads'][lead_name]["Signal"]
        data[case_id]['Leads'][lead_name]['Signal'] = norm_ecg( fix_line(new_entry), mean, sigma ) # modify



def Create_data_generator(data,count_augmentation_data,count_of_diffrent_signals,size_of_data,flag,flag2):
    # generator for learning data with augmentation
    rd.seed(10)
    length_of_move = 10 # step of moving
    SIZE = 5000
    procent_train = 0.8

    count_of_train = int(len(data.keys())*procent_train)
    data_train = { k: data[k] for k in list(data.keys())[:count_of_train]} 
    data_test =  { k: data[k] for k in list(data.keys())[count_of_train:]}

    def modify(data_train, data_test, flag):
        if flag == "healthy":
            wrong_dataset1 = modify_dataset_on_healthy(data_train)
            wrong_dataset2 = modify_dataset_on_healthy(data_test)
            dell_some_ecg(data_train, wrong_dataset1)
            dell_some_ecg(data_test, wrong_dataset2)
        elif flag == "is_not_healthy":
            wrong_dataset1 = modify_dataset_on_sick(data_train)
            wrong_dataset2 = modify_dataset_on_sick(data_test)
            dell_some_ecg(data_train, wrong_dataset1)
            dell_some_ecg(data_test, wrong_dataset2)
        elif flag == None:
            modify_dataset(data_train)
            modify_dataset(data_test)

    modify(data_train, data_test, flag2) 

    if flag == "train":
        DATA = data_train
        print("size of dataset train is ", len(data_train))
    elif flag == "test":
        DATA = data_test
        print("size of dataset test is ", len(data_test))

    while True:
        RES = []
        count = 0
        for i in range(count_of_diffrent_signals):
            case_id = str(rd.sample(DATA.keys(), 1)[0])
            leads = DATA[case_id]["Leads"] # take random a patient
            diagnos = DATA[case_id]['StructuredDiagnosisDoc']
            otvedenie = 'i' # special
            signal = leads[otvedenie]["Signal"] # take a signal 
            start = rd.randint(0,SIZE - size_of_data-count_augmentation_data*length_of_move)
            for x in range(count_augmentation_data+1): #делаем срезы по каждому пациенту
                res = signal[start+x*length_of_move : start+x*length_of_move + size_of_data] # make a slice
                RES.append(res) # add resault in batch
                count +=1
        RES = np.array(RES)
        RES = np.reshape(RES, (count,size_of_data,1))
        yield (RES,RES)

def visualize_latent_space(start, end, decoder, count_of_step, size_of_data):
    from matplotlib.animation import FuncAnimation
    interpol_arr = []
    direction = (end-start)/count_of_step
    RES = []
    for i in range(count_of_step+1):
        tmp = start + i*direction
        RES.append(tmp)
    RES = np.array(RES) # this is our batch

    final_out = decoder.predict(RES)

    fig = plt.figure(3)
    ax1 = fig.add_subplot(1, 1, 1)
    def animate(i):
        x = np.arange(0,size_of_data)
        y = final_out[i].reshape(size_of_data)
        ax1.clear()
        ax1.plot(x, final_out[0])
        ax1.plot(x, final_out[-1])
        ax1.plot(x, y, color = 'k')
        ax1.legend(['ecg A', 'ecg B', 'latent point'], loc='upper left')
        interpol_arr.append(y)
        print('signal was added')
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title("iteration "+ str(i)+ "/"+ str(count_of_step+1))
    anim = FuncAnimation(fig, animate,frames=count_of_step+1, interval=30)
    anim.save('animation_1.gif', writer='imagemagick', fps=60)
    with open('interpol_array.pickle', 'wb') as q:
        pickle.dump(interpol_arr, q)
    plt.show()

def visualize_learning(history, graph1, graph2 ):
    plt.plot(history.history[graph1])
    plt.plot(history.history[graph2])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lern_rate = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lern_rate.append( K.eval(self.model.optimizer.lr) ) 

class Reconstr_on_each_epoch(keras.callbacks.Callback):
    def __init__(self, signal_reconst):
        self.signal = signal_reconst # shape is (2,2000,1)
    def on_train_begin(self, logs={}):
        self.arr = []

    def on_epoch_end(self, epoch, logs=None):
        output = self.model.predict(self.signal)[0]
        self.arr.append(output)


###################################################
## Functions for create some models
##################################################
def create_encoder(input_for_encoder, latent_dim):
    '''create an encoder'''
    x = Conv1D(30, 100, activation='relu', padding='same')(input_for_encoder)
    x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(20, 100, activation='relu', padding='same')(x)
    x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(20, 30, activation='relu', padding='same')(x)
    x =  MaxPooling1D(2, padding='same')(x)
    x = Conv1D(15, 20, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    shape = K.int_shape(x) # save this shape for purposes of reconstruction 
    x = Flatten()(x)
    encoded = Dense(latent_dim, )(x) # закодированный вектор
    return (Model(input_for_encoder, encoded), shape) # create a model 

def create_decoder(input_for_decoder, shape):
    '''create a decoder'''
    x = Dense(shape[1]*shape[2])(input_for_decoder)
    x = Reshape((shape[1] ,shape[2]))(x) 
    x = UpSampling1D(2)(x)
    x = Conv1D(15, 20, activation = 'relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(20, 30, activation = 'relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(20, 100, activation = 'relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 100, padding='same')(x)
    return Model(input_for_decoder, decoded) # create a model

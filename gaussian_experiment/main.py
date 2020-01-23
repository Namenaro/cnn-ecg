import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import data_for_experiment as dfe
import ntpath
ECG_LEN = 5000


def open_json(link):
    f = open(link, 'r')
    data = json.load(f)
    return data


def create_mask(start, end):
    result = np.ones((1, ECG_LEN, 1))
    result[:, start:end, :] = np.zeros((1, end - start, 1))

    return result


def data_generator(some_signal):
    v6_data = np.array(some_signal)
    v6_data = np.expand_dims(v6_data, axis=0)

    while True:
        yield v6_data


def build_model(latent_pool_size, x):
    ae_filters = {
        "conv1": tf.Variable(tf.truncated_normal([100, 1, 10], stddev=0.1)),
        "b_conv1": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
        "conv2": tf.Variable(tf.truncated_normal([80, 10, 10], stddev=0.1)),
        "b_conv2": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
        "conv3": tf.Variable(tf.truncated_normal([50, 10, 5], stddev=0.1)),
        "b_conv3": tf.Variable(tf.truncated_normal([5], stddev=0.1)),

        "deconv3": tf.Variable(tf.truncated_normal([50, 10, 5], stddev=0.1)),
        "b_deconv3": tf.Variable(tf.truncated_normal([5], stddev=0.1)),
        "deconv4": tf.Variable(tf.truncated_normal([80, 10, 10], stddev=0.1)),
        "b_deconv4": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
        "deconv5": tf.Variable(tf.truncated_normal([100, 1, 10], stddev=0.1)),
        "b_deconv5": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
    }

    x_1 = tf.nn.conv1d(x, ae_filters["conv1"], stride=1, padding="SAME", use_cudnn_on_gpu=True) \
          + ae_filters["b_conv1"]
    x_1_mp = tf.layers.max_pooling1d(x_1, pool_size=2, strides=2, padding='SAME')
    x1_relu = tf.nn.leaky_relu(x_1_mp, alpha=0.2)

    x_2 = tf.nn.conv1d(x1_relu, ae_filters["conv2"], stride=1, padding="SAME", use_cudnn_on_gpu=True) \
          + ae_filters["b_conv2"]
    x_2_mp = tf.layers.max_pooling1d(x_2, pool_size=latent_pool_size, strides=latent_pool_size, padding='SAME')
    x2_relu = tf.nn.leaky_relu(x_2_mp, alpha=0.2)

    x_3 = tf.nn.conv1d(x2_relu, ae_filters["conv3"], stride=1, padding="SAME", use_cudnn_on_gpu=True) \
          + ae_filters["b_conv3"]
    x3_mp = tf.layers.max_pooling1d(x_3, pool_size=2, strides=2, padding='SAME')
    x3_relu = tf.nn.leaky_relu(x3_mp, alpha=0.2)

    x_dec1 = tf.contrib.nn.conv1d_transpose(x3_relu, ae_filters["deconv3"],
                                            [1, ECG_LEN // 2 // latent_pool_size + 1, 10],
                                            strides=2, padding="SAME")
    x_dec1_relu = tf.nn.leaky_relu(x_dec1, alpha=0.2)

    x_dec2 = tf.contrib.nn.conv1d_transpose(x_dec1_relu, ae_filters["deconv4"],
                                            [1, ECG_LEN // 2, 10],
                                            strides=latent_pool_size, padding="SAME")
    x_dec2_relu = tf.nn.leaky_relu(x_dec2, alpha=0.2)

    x_dec3 = tf.contrib.nn.conv1d_transpose(x_dec2_relu, ae_filters["deconv5"],
                                            [1, ECG_LEN, 1],
                                            strides=2, padding="SAME")
    x_dec3_relu = tf.nn.leaky_relu(x_dec3, alpha=0.2)

    return x_dec3_relu


def fit(mask_start, mask_end, epochs, data_to_fit, param, config):
    error_inside_list = []
    error_outside_list = []
    error_inside = 0
    error_outside = 0

    generator = data_generator(data_to_fit)
    mask = create_mask(mask_start, mask_end)
    inversed_mask = np.ones(mask.shape) - mask

    tf.reset_default_graph()

    x_ph = tf.placeholder(tf.float32, [None, ECG_LEN])
    x = tf.reshape(x_ph, [-1, ECG_LEN, 1])
    model_output = build_model(param, x)

    vars = tf.trainable_variables()
    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 1

    error_function = tf.reduce_mean(tf.square(model_output - x) * mask) + loss_l2
    error_function_outside = tf.reduce_mean(tf.square(model_output - x) * mask)
    error_function_inside = tf.reduce_mean(tf.square(model_output - x) * inversed_mask)

    optimizer = tf.train.AdagradOptimizer(0.01).minimize(error_function)

    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:

        sess.run(init)

        for epoch in range(epochs):
            epoch_x = next(generator)
            _, _, error_inside, error_outside = sess.run(
                [optimizer, error_function, error_function_inside, error_function_outside], feed_dict={x_ph: epoch_x})

            error_inside_list.append(error_inside)
            error_outside_list.append(error_outside)

    return error_inside, error_outside

def make_experiment(epochs, start_mask, end_mask, name):
    folder = os.getcwd()


    os.chdir(folder)
    ecgs, unperiodic, ids = dfe.restore_experimental_data()

    print("data extracted")

    config = tf.ConfigProto(device_count={'GPU': 1})

    columns = ['in ecg', 'out ecg']
    df_good_ecg = pd.DataFrame(columns=columns, index=ids)
    df_bad_ecg = pd.DataFrame(columns=columns, index=ids)

    for i in range(len(ids)-1):
        # for everi signal we try good and bad
        for param in [3, 7]:
            data_to_fit = ecgs[i]
            print(str(i) + "-th signal:")
            error_inside, error_outside = fit(start_mask, end_mask, epochs, data_to_fit, param, config)
            print(error_inside, error_outside)

            if (param == 3):
                df_good_ecg.at[ids[i], 'in ecg'] = error_inside
                df_good_ecg.at[ids[i], 'out ecg'] = error_outside
            else:
                df_bad_ecg.at[ids[i], 'in ecg'] = error_inside
                df_bad_ecg.at[ids[i], 'out ecg'] = error_outside
    df_good_ecg.to_csv(name+ "_good_table_ECG.csv", sep='\t')
    df_bad_ecg.to_csv(name+ "_bad_table_ECG.csv", sep='\t')

    ########################################################
    #### now - unperiodics stochastic signal ###############
    ########################################################
    columns = ['in zone', 'out zone']
    df_good_un = pd.DataFrame(columns=columns, index=ids)
    df_bad_un = pd.DataFrame(columns=columns, index=ids)
    for i in range(len(ids)-1):
        # for everi signal we try good and bad
        for param in [3, 7]:
            data_to_fit = unperiodic[i]
            print( str(i) + "-th signal:")
            error_inside, error_outside = fit(start_mask, end_mask, epochs, data_to_fit, param, config)
            print(error_inside, error_outside)

            if (param == 3):
                df_good_un.at[ids[i], 'in zone' ] = error_inside
                df_good_un.at[ids[i], 'out zone'] = error_outside
            else:
                df_bad_un.at[ids[i], 'in zone'] = error_inside
                df_bad_un.at[ids[i], 'out zone'] = error_outside

    df_good_un.to_csv(name+"_good_table_unperiodic.csv", sep='\t')
    df_bad_un.to_csv(name+ "_bad_table_unperiodic.csv", sep='\t')

if __name__ == "__main__":
    make_experiment(epochs=1000, start_mask=500, end_mask=2000, name="one")
    make_experiment(epochs=500, start_mask=500, end_mask=2000, name="two")
    make_experiment(epochs=1000, start_mask=1000, end_mask=2000, name="3")
    make_experiment(epochs=500, start_mask=1000, end_mask=2000, name="4")
    make_experiment(epochs=2000, start_mask=1000, end_mask=4900, name="5")
    make_experiment(epochs=2000, start_mask=1000, end_mask=4900, name="6")
    make_experiment(epochs=2000, start_mask=200, end_mask=4900, name="7")
    make_experiment(epochs=2000, start_mask=400, end_mask=4900, name="8")
    make_experiment(epochs=4000, start_mask=600, end_mask=4900, name="8")
    make_experiment(epochs=4000, start_mask=600, end_mask=1600, name="10")
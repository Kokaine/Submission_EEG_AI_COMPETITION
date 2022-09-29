from os import name
from tcn import TCN, tcn_full_summary
from tensorflow.python.keras.layers import Dense, Dropout, Softmax
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import Model
import scipy.io as scio
import numpy as np
import tensorflow as tf

#Processing Raw data
data_mat = scio.loadmat('data_EEG_AI.mat')
raw_data = np.array(data_mat['data'])
channels = np.array(data_mat['channel_labels'])
label_mat = scio.loadmat('label 7800x1.mat')
raw_label = np.array(label_mat['label'])
raw_label -= 1

#batch_size, time_steps, input_dim = 64, 251, 1

mean_data = raw_data.mean(2)

for i in range(7800):
    raw_data[:, :, i] -= mean_data
used_data = raw_data.copy()
for i in range(24):
    for j in range(50,301):
        raw_data[i, j, :] /= np.std(raw_data[i, j, :])
print(np.shape(raw_data), np.shape(raw_label))

# breakpoint()

output_softmax = np.empty([24, 1300, 26])
for jj in range(24):
    print(jj)
    # divide dataset into training and testing set
    data_train_X = np.empty([1, 251, 6500])
    data_test_X = np.empty([1, 251, 1300])
    data_train_Y = np.zeros([6500])
    data_test_Y = np.zeros([1300])
    seq = [6,8,9,10,11,15,20,23]
    seq1 = [0,1,2,3,4,5,7,12,13,14,16,17,18,19,21,22]
    seq2 = [0, 1, 3, 5, 8]


    for i in range(26):
        data_train_X[:, :, 250*i : 250*i + 250] = raw_data[jj, 50:301, 300*i : 300*i + 250]
        data_test_X[:, :, 50*i : 50*i + 50] = raw_data[jj, 50:301, 300*i+250 : 300*i+300]
        data_train_Y[250*i : 250*i+250] = raw_label[300*i : 300*i+250, 0]
        data_test_Y[50*i : 50*i+50] = raw_label[300*i+250 : 300*i+300, 0]

    data_train_X = data_train_X.transpose(2,0,1)
    data_test_X = data_test_X.transpose(2,0,1)

    def get_x_y(size=7800):
        return data_train_X, data_train_Y

    tcn_layer1 = TCN(input_shape=(1, 251),
            nb_filters=256,
            kernel_size=3,
            nb_stacks=3,
            dilations=(1, 2, 4, 8),
            padding='causal',
            use_skip_connections=True,
            dropout_rate=0.2,
            return_sequences=False,
            activation='tanh',
            kernel_initializer='he_normal',
            use_batch_norm=True,
            use_layer_norm=False,
            use_weight_norm=False,
            )
    # The receptive field tells you how far the model can see in terms of timesteps.
    print('Receptive field size =', tcn_layer1.receptive_field)

    m = Sequential([
        tcn_layer1,
        Dense(26, activation='softmax', name = 'Softmax')
        #softmax
    ])
    #构建从输入层（第0层）到第1层的子模型
    #partialmodel = tf.keras.Model(m.inputs, m.layers.output)

    #x = np.random.rand(...) #测试用的输入
    #output_train = partialmodel([x], training=True)   # runs the model in training mode
    #output_test = partialmodel([x], training=False)   # runs the model in test mode

    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #tcn_full_summary(m, expand_residual_blocks=False)
    tensorboard = TensorBoard(
        log_dir='logs_tcn',
        histogram_freq=1,
        write_images=True
    )
    x, y = get_x_y()


    m.fit(x, y,
        batch_size=128,
        validation_data=(data_test_X, data_test_Y),
        callbacks=[tensorboard],
        epochs=20)
    output = Model(inputs=m.input,outputs=m.get_layer('Softmax').output)
    output_softmax[jj, :, :] = output.predict(data_test_X)#这个是x_train或者x_test

    m.summary()
    m.save_weights('model_data/model_weight'+str(jj)+'.h5')
    scio.savemat('model_data/output_softmax'+str(jj)+'.mat', mdict={'output_softmax'+str(jj):output_softmax[jj, :, :]})
    print('channel'+str(jj)+'saved successfully!!!')
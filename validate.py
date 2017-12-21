from sklearn.metrics import roc_auc_score

import time
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input, Embedding
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from dataset_manager import DatasetManager

import pandas as pd
import numpy as np

import sys
from sys import argv

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt


def create_and_evaluate_model(args):
    # compile a model with same parameters that was trained, and load the weights of the trained model
    print('Training model...')
    start = time.time()

    dropout = args['dropout']
    lstmsize = int(args['lstmsize'])
    learning_rate = args['learning_rate']
    batch_size = int(args['batch_size'])
    n_layers = int(args["n_layers"])
    
    main_input = Input(shape=(max_len, data_dim), name='main_input')

    if n_layers == 1:
        l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(main_input)
        b_last = BatchNormalization()(l1)

    elif n_layers == 2:
        l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(main_input)
        b1 = BatchNormalization(axis=1)(l1)
        l2 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(b1)
        b_last = BatchNormalization()(l2)

    elif n_layers == 3:
        l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(main_input)
        b1 = BatchNormalization(axis=1)(l1)
        l2 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(b1)
        b2 = BatchNormalization(axis=1)(l2)
        l3 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(b2)
        b_last = BatchNormalization()(l3)

    outcome_output = Dense(2, activation=activation, kernel_initializer='glorot_uniform', name='outcome_output')(b_last)

    model = Model(inputs=[main_input], outputs=[outcome_output])
    if optimizer == "adam":
        opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    elif optimizer == "rmsprop":
        opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(loss={'outcome_output':'binary_crossentropy'}, optimizer=opt)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    history = model.fit({'main_input': X}, {'outcome_output':y_o}, validation_data=(X_val, y_o_val), verbose=2, callbacks=[early_stopping, lr_reducer], batch_size=batch_size, epochs=nb_epoch)

    print("Done: %s"%(time.time() - start))
    
    pred_y_o = model.predict(X_val, verbose=0)
    print(set(y_o_val[:,0]))
    score = roc_auc_score(y_o_val[:,0], pred_y_o[:,0])

    print('Val AUC:', score)
    sys.stdout.flush()
    return {'loss': -score, 'status': STATUS_OK, 'model': model}


print('Preparing data...')
start = time.time()

dataset_name = argv[1]
embedding_type = argv[2]
embedding_dim = int(argv[3])

scale_model = "row"

train_ratio = 0.8
val_ratio = 0.2
activation = "sigmoid"
optimizer = "adam"
nb_epoch = 50

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
train, val = dataset_manager.split_val(train, val_ratio, split="random")

if embedding_type == "none":
    dt_train = dataset_manager.encode_data_with_label_all_data(train)
    dt_val = dataset_manager.encode_data_with_label_all_data(val)
else:
    dt_train = dataset_manager.encode_data_with_label_all_data_act_res_embedding(train, embedding_type=embedding_type, embedding_dim=embedding_dim,  scale_model=scale_model)
    dt_val = dataset_manager.encode_data_with_label_all_data_act_res_embedding(val, embedding_type=embedding_type, embedding_dim=embedding_dim, scale_model=scale_model)

if "bpic2017" in dataset_name:
    max_len = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.95))
else:
    max_len = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.95))

data_dim = dt_train.shape[1] - 3

X, _, _, y_o = dataset_manager.generate_3d_data_with_label_all_data(dt_train, max_len)
X_val, _, _, y_o_val = dataset_manager.generate_3d_data_with_label_all_data(dt_val, max_len)

print(X.shape, y_o.shape, X_val.shape, y_o_val.shape)

print("Done: %s"%(time.time() - start))


print('Optimizing parameters...')
space = {
        'dropout': hp.uniform('dropout', 0, 0.3),
        'lstmsize': hp.choice('lstmsize', [str(val) for val in range(10, 151)]),
        'batch_size': hp.choice('batch_size', ["8", "16", "32", "64"]),
        'n_layers': hp.choice('n_layers', ["1", "2", "3"]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.000001), np.log(0.0001))
    }

trials = Trials()
best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=8, trials=trials)

print best
print hyperopt.space_eval(space, best)
for trial in trials.trials:
    print trial
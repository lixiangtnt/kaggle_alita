from sklearn.model_selection import train_test_split
from sklearn import linear_model
import data_utils
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply, subtract
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.noise import GaussianNoise
from keras.layers import Bidirectional
import os
import time

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

train_X, train_Y, test_X = data_utils.load_data()
X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.1, random_state=42)
X_train = X_train.values
X_valid = X_valid.values

# model
len_feat = X_train.shape[1]
print(len_feat)
input1 = Input(shape=(len_feat,), dtype="float32")
# features_dense = BatchNormalization()(features_dense)
features_dense = Dense(500, activation="relu")(input1)
# features_dense = Dropout(0.2)(features_dense)
features_dense = Dense(100, activation="relu")(features_dense)
out = Dense(1, activation="sigmoid")(features_dense)
model = Model(inputs=input1, outputs=out)
model.compile(loss="binary_crossentropy",
              optimizer="nadam", metrics=["accuracy"])

best_model_path = os.path.abspath(os.path.join(out_dir, "checkpoints", "model"))+"/best_model" + ".h5"

os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=False)

# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
#                           write_graph=True, write_images=False)  # adding tensorboard in callbacks
hist = model.fit([X_train], Y_train,
                 validation_data=([X_valid], Y_valid),
                 epochs=15, batch_size=32, shuffle=True,
                 callbacks=[model_checkpoint], verbose=1)

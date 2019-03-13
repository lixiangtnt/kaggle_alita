from sklearn.model_selection import train_test_split
from sklearn import linear_model
import data_utils
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply, subtract
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.noise import GaussianNoise
from keras.layers import Bidirectional
from keras import backend as K
import os
import time

timestamp = str(int(time.time()))
# timestamp = "190310"
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

train_X, train_Y, test_X = data_utils.load_data()
X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.1, random_state=42)
X_train = X_train.values
X_valid = X_valid.values

def train():
    def expand_dims(x):

        return K.expand_dims(x, axis=-1)

    # model
    len_feat = X_train.shape[1]
    print(len_feat)
    input1 = Input(shape=(len_feat,), dtype="float32")
    #
    # features_dense = BatchNormalization()(input1)
    # features_dense = GaussianNoise(0.1)(features_dense)

    lstm_layer = Bidirectional(LSTM(75, recurrent_dropout=0.2, return_sequences=True))
    input2 = Lambda(expand_dims)(input1)
    lstm_out = lstm_layer(input2)
    lstm_ave = GlobalAveragePooling1D()(lstm_out)
    lstm_max = GlobalMaxPooling1D()(lstm_out)
    # lstm_full = concatenate([])
    features_dense = Dense(1000, activation="relu")(input1)
    features_dense = Dropout(0.1)(features_dense)
    features_dense = Dense(100, activation="relu")(features_dense)
    features_dense = concatenate([features_dense, lstm_ave, lstm_max])
    features_dense = BatchNormalization()(features_dense)
    features_dense = GaussianNoise(0.1)(features_dense)
    features_dense = Dropout(0.2)(features_dense)
    features_dense = BatchNormalization()(features_dense)
    out = Dense(1, activation="sigmoid")(features_dense)

    model = Model(inputs=input1, outputs=out)
    model.compile(loss="binary_crossentropy",
                  optimizer="RMSProp", metrics=["accuracy"])

    best_model_path = os.path.abspath(os.path.join(out_dir, "checkpoints", "model", "best_model_val_{val_acc:.3f}.h5"))#+"/best_model" + ".h5"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=False)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)  # adding tensorboard in callbacks
    hist = model.fit([X_train], Y_train,
                     validation_data=([X_valid], Y_valid),
                     epochs=15, batch_size=16, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint], verbose=1)

    for a, b, c in os.walk(os.path.dirname(best_model_path)):
        best_model_path = os.path.join(a, c[0])

    model = load_model(best_model_path)
    preds = model.predict([X_valid], batch_size=32, verbose=1)
    submission = pd.DataFrame({"ground_truth": Y_valid, "result": preds.ravel()})
    from sklearn.metrics import auc, roc_curve
    valid_auc = ""
    if submission["ground_truth"].shape == submission["result"].shape:
        fpr, tpr, thresholds = roc_curve(submission['ground_truth'], submission['result'], pos_label=1)
        valid_auc = auc(fpr, tpr)
    print("_auc_{}".format(valid_auc))
    submission.to_csv(best_model_path+"_auc_{:.4f}".format(valid_auc)+"_result.csv", sep="\t", index=False)


def pred(model_path):
    model = load_model(model_path)
    test_dl = test_X.iloc[:,1:].values
    preds = model.predict([test_dl], batch_size=32, verbose=1)
    submission = pd.DataFrame({"ID_code": test_X["ID_code"], "target": preds.ravel()})
    # from sklearn.metrics import auc, roc_curve
    # valid_auc = ""
    # if submission["ground_truth"].shape == submission["result"].shape:
    #     fpr, tpr, thresholds = roc_curve(submission['ground_truth'], submission['result'], pos_label=1)
    #     valid_auc = auc(fpr, tpr)
    # print("_auc_{}".format(valid_auc))
    submission.to_csv(model_path+"_submission.csv", index=False)


if __name__=="__main__":
    train()
    # pred("runs/1552231586/checkpoints/model/best_model_val_0.909.h5")



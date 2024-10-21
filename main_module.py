import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os
import sys
from IPython.display import display, HTML
from itertools import combinations, chain

class SuppressOutput:
    def __enter__(self):
        self.stdout_original = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.stdout_original

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Normalized Observed log(TX/Txref)'] = MinMaxScaler().fit_transform(df[['Observed log(TX/Txref)']])
    return df

def combine_columns(df):
    X = df[['Promoter Sequence']].astype(str).agg(''.join, axis=1)
    y = df['Normalized Observed log(TX/Txref)']
    return X, y

def padded_one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], '0': [0,0,0,0]}
    encoding = [mapping[nucleotide.upper()] for nucleotide in sequence]
    return encoding

def preprocess_sequences(X, max_length=None):
    if max_length is None:
        max_length = max(len(seq) for seq in X)
    padded_sequences = [padded_one_hot_encode('0' * (max_length - len(seq)) + seq) for seq in X]
    return np.array(padded_sequences), max_length

def get_training_data(file_path):

    df = load_and_preprocess_data(file_path)
    X, y = combine_columns(df)
    X, max_length = preprocess_sequences(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def decode_to_df(encoded_sequence):
    mapping = {(1, 0, 0, 0): 'A', (0, 1, 0, 0): 'T', (0, 0, 1, 0): 'C', (0, 0, 0, 1): 'G', (0, 0, 0, 0): '0'}
    data = []
    for seq in encoded_sequence:
        data.append(''.join([mapping[tuple(nucleotide)] for nucleotide in seq]))
    return pd.DataFrame(data, columns=['Promoter Sequence'])

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    with SuppressOutput():
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history

def evaluate_model(model, X_test, y_test):
    with SuppressOutput():
        loss = model.evaluate(X_test, y_test)
    return loss

def save_model(model, filename):
    model.save(filename)

def load_and_predict(filename, X):
    model = load_model(filename)
    predictions_array = model.predict(np.array(X))[:, 0]
    return pd.DataFrame(predictions_array, columns=['Value'])

def run(X_train, y_train, X_test, y_test, model_name, epochs=100, batch_size=32):
    model = build_cnn_model(X_train.shape[1:])
    history = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size)
    loss = evaluate_model(model, X_test, y_test)

    return model, history, loss

def split_data_by_file(df):

    # Get the unique file names
    file_names = df['File Name'].unique()

    # Split the data by file
    split_data = {name: {} for name in file_names}
    split_data['all'] = {}

    # Preprocess the data, split for each file
    for file in file_names:
        filtered_df = df[df['File Name'].isin([file])]
        X, y = combine_columns(filtered_df)
        X, max_length = preprocess_sequences(X, 150)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        split_data[file]['X_train'] = X_train
        split_data[file]['X_test'] = X_test
        split_data[file]['y_train'] = y_train
        split_data[file]['y_test'] = y_test

    # Preprocess the data, include all files
    X, y = combine_columns(df)
    X, max_length = preprocess_sequences(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    split_data['all']['X_train'] = X_train
    split_data['all']['X_test'] = X_test
    split_data['all']['y_train'] = y_train
    split_data['all']['y_test'] = y_test

    return split_data, file_names

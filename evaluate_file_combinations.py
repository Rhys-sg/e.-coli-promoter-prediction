import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from itertools import combinations, chain

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
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

def evaluate_model(model, X_test, y_test):
    return model.evaluate(X_test, y_test)

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


def evaluate_file_combinations(file_names, split_data):

    # Generate all combinations of file_names
    all_combinations = list(chain.from_iterable(combinations(file_names, i) for i in range(1, len(file_names) + 1)))

    # Store the number of files and corresponding loss for plotting
    data_for_plot = []

    # Create a model for each combination of files
    for i, file_combo in enumerate(all_combinations):
        print(f'Training model {i}/{len(all_combinations)} with files: {file_combo}', end='\r')
        combined_X_train = np.concatenate([split_data[file_name]['X_train'] for file_name in file_combo], axis=0)
        combined_y_train = np.concatenate([split_data[file_name]['y_train'] for file_name in file_combo], axis=0)
        
        # Convert file_combo to a string, e.g., 'file1, file2' instead of ('file1', 'file2')
        file_combo_str = ', '.join(file_combo)
        
        # Make model and get MSE
        model, history, loss = run(combined_X_train,
                                   combined_y_train,
                                   split_data['all']['X_test'],
                                   split_data['all']['y_test'],
                                   file_combo_str)
        
        # Record the number of files and the loss
        data_for_plot.append((len(file_combo), np.log(loss), file_combo_str))

    return data_for_plot

def save_data_for_plot_to_csv(data_for_plot, filename='data_comparison.csv'):
    df = pd.DataFrame(data_for_plot)
    df.columns = ['Number of Files', 'MSE', 'Name']
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    file_path = 'LaFleur_supp.csv'
    df = load_and_preprocess_data(file_path)
    split_data, file_names = split_data_by_file(df)
    data_for_plot = evaluate_file_combinations(file_names, split_data)
    save_data_for_plot_to_csv(data_for_plot)
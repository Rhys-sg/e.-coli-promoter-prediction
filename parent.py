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

def reshape_model_input(X):
    return np.array([[x, x, x, x] for x in X.values]).reshape(-1, 1, 4)

def concatenate_inputs(array1, array2):
    return np.concatenate((array1, array2), axis=1)

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
    if not filename.endswith('.keras'):
        filename += '.keras'
    model.save('models/' + filename)

def load_and_predict(filename, X):
    if not filename.endswith('.keras'):
        filename += '.keras'
    model = load_model('../Models/' + filename)
    predictions_array = model.predict(np.array(X))[:, 0]
    return pd.DataFrame(predictions_array, columns=['Value'])

def run(X_train, y_train, X_test, y_test, model_name, epochs=5, batch_size=32):
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

def evalue_each_file(file_names, split_data, preceding_message=''):

    loss_values = {name: [] for name in file_names}    

    # Create a model for each file
    for i, file in enumerate(file_names):
        print(f'{preceding_message}Training Model {i + 1}/{len(file_names)}', end='\r')

        # Train the model on the filtered data
        model, history, loss = run(split_data[file]['X_train'],
                                        split_data[file]['y_train'],
                                        split_data[file]['X_test'],
                                        split_data[file]['y_test'],
                                        file)
        loss_values[file].append(loss)

        # Train the model on all the data
        model, history, loss = run(split_data[file]['X_train'], 
                                   split_data[file]['y_train'],
                                   split_data['all']['X_test'],
                                   split_data['all']['y_test'],
                                   file)
        
        loss_values[file].append(loss)
    
    return loss_values

def repeat_evalute_each_file(file_names, split_data, iterations):
    all_results = {name: [] for name in file_names}
    
    # Evaluate each file and append the MSE from each iteration
    for i in range(iterations):
        loss_values = evalue_each_file(file_names, split_data, f'Iteration {i+1}/{iterations}, ')
        for file in file_names:
            all_results[file].append(loss_values[file])
    
    # Convert the results to averages over the X runs
    averaged_results = {file: np.mean(all_results[file], axis=0) for file in file_names}
    
    return all_results, averaged_results

def plot_repeat_evalute_each_file(file_names, all_results, averaged_results):
    # Sort file_names based on "All Data MSE" (second element in averaged_results)
    sorted_file_names = sorted(file_names, key=lambda file: averaged_results[file][1])
    
    # Prepare data
    x = np.arange(len(sorted_file_names))
    bar_width = 0.35  # Adjust the width for two columns

    # Extracting the individual and average results
    training_data_mse = {file: [result[0] for result in all_results[file]] for file in sorted_file_names}
    all_data_mse = {file: [result[1] for result in all_results[file]] for file in sorted_file_names}

    avg_training_data = [averaged_results[file][0] for file in sorted_file_names]
    avg_all_data = [averaged_results[file][1] for file in sorted_file_names]

    # Creating the plot
    plt.figure(figsize=(10, 6))

    # Scatter plot for each file
    for i, file in enumerate(sorted_file_names):
        plt.scatter([x[i] - bar_width / 2] * len(training_data_mse[file]), training_data_mse[file], color='skyblue', label='Training Data MSE' if i == 0 else "")
        plt.scatter([x[i] + bar_width / 2] * len(all_data_mse[file]), all_data_mse[file], color='lightgreen', label='All Data MSE' if i == 0 else "")
        
        # Add horizontal line for Training Data MSE average and All Data MSE average
        plt.hlines(avg_training_data[i], x[i] - bar_width / 2 - 0.05, x[i] - bar_width / 2 + 0.05, colors='grey', linestyles='solid', label='Avgerage MSE' if i == 0 else "")
        plt.hlines(avg_all_data[i], x[i] + bar_width / 2 - 0.05, x[i] + bar_width / 2 + 0.05, colors='grey', linestyles='solid', label='')

    # Formatting
    plt.xticks(x, sorted_file_names, rotation=45, ha='right', rotation_mode='anchor')
    plt.xlabel('Files')
    plt.ylabel('Mean Squared Error (MSE)')

    # Adjust legend and layout
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

def save_loss_values_to_csv(loss_values, filename='loss_values.csv'):
    df = pd.DataFrame.from_dict(loss_values, orient='index', columns=['Training Data MSE', 'All Data MSE'])
    df.to_csv(filename)

def read_loss_values_from_csv(filename='loss_values.csv'):
    df = pd.read_csv(filename, index_col=0)
    loss_values = df.to_dict(orient='index')
    return {key: list(value.values()) for key, value in loss_values.items()}

def plot_each_file_MSE_stacked(file_names, loss_values):
    # Sort file_names based on the "All Data MSE" (second element in each loss_values entry)
    sorted_file_names = sorted(file_names, key=lambda file: loss_values[file][1])

    # Prepare the data for plotting
    bar_width = 0.35
    x = np.arange(len(sorted_file_names))

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    training_bars = plt.bar(x - bar_width/2, [loss_values[file][0] for file in sorted_file_names], width=bar_width, label='Training Data MSE', color='skyblue')
    all_data_bars = plt.bar(x + bar_width/2, [loss_values[file][1] for file in sorted_file_names], width=bar_width, label='All Data MSE', color='lightgreen')

    # Add values above the bars
    for bar in training_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', va='bottom', ha='center', fontsize=8)

    for bar in all_data_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', va='bottom', ha='center', fontsize=8)

    # Add labels and title
    plt.xlabel('Training Data')
    plt.ylabel('MSE')

    # Adjust the x-axis labels to align to the right (so the end of each label points to the tick)
    plt.xticks(x, sorted_file_names, rotation=45, ha='right', rotation_mode='anchor')

    plt.legend()
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('images/MSE for training vs all data.png')
    plt.show()

def plot_each_file_MSE_separate(file_names, loss_values):
    # Sort file_names based on the "All Data MSE" (second element in each loss_values entry)
    sorted_file_names = sorted(file_names, key=lambda file: loss_values[file][1])

    # Prepare the data for plotting
    x = np.arange(len(sorted_file_names))
    training_mse = [loss_values[file][0] for file in sorted_file_names]
    all_data_mse = [loss_values[file][1] for file in sorted_file_names]

    # Plot the Training Data MSE bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(x, training_mse, color='skyblue')
    plt.xlabel('Training Data')
    plt.ylabel('Training Data MSE')

    # Adjust the x-axis labels to align to the right (so the end of each label points to the tick)
    plt.xticks(x, sorted_file_names, rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    plt.show()

    # Plot the All Data MSE bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(x, all_data_mse, color='lightgreen')
    plt.xlabel('Training Data')
    plt.ylabel('All Data MSE')

    # Adjust the x-axis labels to align to the right (so the end of each label points to the tick)
    plt.xticks(x, sorted_file_names, rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.show()


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

def read_data_for_plot_from_csv(filename='data_comparison.csv'):
    return pd.read_csv(filename).values

def plot_each_file(file_names, loss_values):
    # Sort file_names based on the "All Data MSE" (second element in each loss_values entry)
    sorted_file_names = sorted(file_names, key=lambda file: loss_values[file][1])

    # Prepare the data for plotting
    bar_width = 0.35
    x = np.arange(len(sorted_file_names))

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - bar_width/2, [loss_values[file][0] for file in sorted_file_names], width=bar_width, label='Training Data MSE', color='skyblue')
    bars2 = plt.bar(x + bar_width/2, [loss_values[file][1] for file in sorted_file_names], width=bar_width, label='All Data MSE', color='lightgreen')

    # Add value labels on top of the bars
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    # Add labels and title
    plt.xlabel('Training Data')
    plt.ylabel('MSE')
    plt.xticks(x, sorted_file_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig('images/all file combinations.png')
    plt.show()

def plot_file_combinations(data_for_plot):

    # Extract data for plotting
    num_files = [entry[0] for entry in data_for_plot]
    loss_values = [entry[1] for entry in data_for_plot]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(num_files, loss_values, color='grey', s=50, alpha=0.5)

    plt.xlabel('Number of Files')
    plt.ylabel('log10(MSE)')

    # Save and show the plot
    plt.savefig('images/all file combinations.png')
    plt.show()

def plot_individual_file_combinations(data_for_plot, file_names):

    # Extract data for plotting
    num_files = [entry[0] for entry in data_for_plot]
    loss_values = [entry[1] for entry in data_for_plot]
    file_combos = [entry[2] for entry in data_for_plot]

    # Create separate scatter plots for each file name
    for file_name in file_names:
        plt.figure(figsize=(10, 6))

        # Separate data based on whether the current file is included or not
        included_points = [(n, l) for (n, l, combo) in data_for_plot if file_name in combo]
        not_included_points = [(n, l) for (n, l, combo) in data_for_plot if file_name not in combo]
        
        # Plot included points in color
        if included_points:
            num_files_included, loss_included = zip(*included_points)
            plt.scatter(num_files_included, loss_included, label=f'Includes {file_name}', color='tab:blue', s=100)
        
        # Plot not included points in grey
        if not_included_points:
            num_files_not_included, loss_not_included = zip(*not_included_points)
            plt.scatter(num_files_not_included, loss_not_included, color='grey', s=50, alpha=0.5)

        # Set plot details
        plt.xlabel('Number of Files')
        plt.ylabel('log10(MSE)')
        plt.legend()

        # Save and show the plot
        plt.savefig(f'images/{file_name} file combinations.png')
        plt.show()


def plot_saliency_map_sequence(sequence, base_px=16, multiplier=1, model_filename='model.keras'):
    """
    Visualizes the saliency map of the CNN model for a single sequence using a bar graph.

    Parameters:
     - model_filename (str): The filename of the model to load.
     - sequence (str): The sequence to visualize.
    """
    
    model = load_model(model_filename)
    encoded_sequence, max_length = preprocess_sequences([sequence], 150)
    
    target_class_index = 0

    saliency_map = generate_saliency_map(model, encoded_sequence[0], target_class_index)[0][max_length - len(sequence):]

    # Color map for nucleotides
    color_map = {
        'T': 'Tomato',
        'G': 'MediumSeaGreen',
        'A': 'orange',
        'C': 'DodgerBlue',
    }
    
    # Generate HTML content with colors
    html_content = ''.join(
        f"<span style='font-size: {(base_px * mag * multiplier) + base_px}px; color: {color_map[bp]};'>{bp}</span>"
        for bp, mag in zip(sequence, saliency_map)
    )

    # Display the content
    display(HTML(html_content))


def plot_saliency_map_grid(model_filename='model.keras', data_filename='LaFleur_supp.csv', i_start=0, i_end=20):
    """
    Visualizes the saliency map of the CNN model for multiple sequences in a grid.
    By default, this uses model.keras, and the first 20 sequences in LaFleur_supp.csv.

    Parameters:
     - model_filename (str): The filename of the model to load.
     - data_filename (str): The filename of the data to load.
     - i_start (int): The starting index of the data to use.
     - i_end (int): The ending index of the data to use.

    """

    model = load_model(model_filename)
    data = pd.read_csv(data_filename).loc[i_start:i_end-1, 'Promoter Sequence']
    data, max_length = preprocess_sequences(data, 150)
    target_class_index = 0
    
    stacked_saliency_map = np.vstack([generate_saliency_map(model, sequence, target_class_index) for sequence in data])
    plt.imshow(stacked_saliency_map, cmap='magma', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def generate_saliency_map(model, sequence, target_class_index):
    input_tensor = tf.convert_to_tensor(sequence[np.newaxis, ...], dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        loss = predictions[0, target_class_index]
        
    gradients = tape.gradient(loss, input_tensor)
    
    # Normalize gradients
    gradients = tf.norm(gradients, axis=-1)
    gradients = gradients / tf.reduce_max(gradients)
    
    return gradients.numpy()
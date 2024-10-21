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


def save_loss_values_to_csv(loss_values, filename='Data/loss_values.csv'):
    df = pd.DataFrame.from_dict(loss_values, orient='index', columns=['Training Data MSE', 'All Data MSE'])
    df.to_csv(filename)

def read_loss_values_from_csv(filename='Data/loss_values.csv'):
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
    plt.savefig('Images/MSE for training vs all data.png')
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

def save_data_for_plot_to_csv(data_for_plot, filename='Data/data_comparison.csv'):
    df = pd.DataFrame(data_for_plot)
    df.columns = ['Number of Files', 'MSE', 'Name']
    df.to_csv(filename, index=False)

def read_data_for_plot_from_csv(filename='Data/data_comparison.csv'):
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
    plt.savefig('Images/all file combinations.png')
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
    plt.savefig('Images/all file combinations.png')
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
        plt.savefig(f'Images/{file_name} file combinations.png')
        plt.show()

def plot_individual_file_combinations_grid(data_for_plot, file_names):
    # Extract data for plotting
    num_files = [entry[0] for entry in data_for_plot]
    loss_values = [entry[1] for entry in data_for_plot]
    file_combos = [entry[2] for entry in data_for_plot]
    
    # Create a 3x3 grid for 9 files
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Reduced spacing
    
    # Loop through each file and plot it in a grid cell
    for i, file_name in enumerate(file_names):
        row, col = divmod(i, 3)  # Determine the row and column in the grid
        ax = axs[row, col]  # Select the corresponding subplot
        
        # Separate data based on whether the current file is included or not
        included_points = [(n, l) for (n, l, combo) in data_for_plot if file_name in combo]
        not_included_points = [(n, l) for (n, l, combo) in data_for_plot if file_name not in combo]
        
        # Plot included points in color
        if included_points:
            num_files_included, loss_included = zip(*included_points)
            ax.scatter(num_files_included, loss_included, label=f'Includes {file_name}', color='tab:blue', s=100)
        
        # Plot not included points in grey
        if not_included_points:
            num_files_not_included, loss_not_included = zip(*not_included_points)
            ax.scatter(num_files_not_included, loss_not_included, color='grey', s=50, alpha=0.5)
        
        # Set plot details for the current subplot
        ax.set_xlabel('Number of Files')
        if i % 3 == 0:
            ax.set_ylabel('log10(MSE)')
        else:
            ax.set_yticklabels([])
        ax.legend()
    
    # Save the entire 3x3 grid of plots
    plt.savefig('Images/Figure 2.png')
    plt.savefig('Images/Figure 2.pdf', format='pdf')
    plt.show()


def get_file_mse_effect(file_names, data_for_plot):
    all_combinations = list(chain.from_iterable(combinations(file_names, i) for i in range(1, len(file_names) + 1)))
    file_mse = {tuple(sorted(each)) : None for each in all_combinations}
    for each in data_for_plot:
        key = tuple(sorted(each[2].split(', ')))
        file_mse[key] = each[1]

    mse_effect = {file : [] for file in file_names}

    for key, value in file_mse.items():
        for file in file_names:
            if file in key:
                continue
            next_key = tuple(sorted(key + (file,)))
            mse_difference = value - file_mse[next_key]
            mse_effect[file].append(mse_difference)
    
    return file_mse, mse_effect

def plot_mse_effect_with_ttest(file_mse, file_names, file_of_interest, start_from=1, end_at=None):
    """
    Plots the distribution of MSE differences for combinations that include
    and exclude a given file of interest. Also performs a t-test for significance.

    Parameters:
    - file_mse: dictionary where keys are tuples of file names and values are MSEs
    - file_names: list of all file names
    - file_of_interest: the name of the file to analyze (e.g., 'Urtecho et al')
    - start_from: minimum number of files in the combination to plot
    - end_at: maximum number of files in the combination to plot
    """
    
    # 1. Initialize lists to store MSE effects
    include_mse_effect = []
    exclude_mse_effect = []
    
    # 2. Calculate MSE differences for combinations including and excluding the file_of_interest
    for key, value in file_mse.items():
        for file in file_names:
            if file in key:
                continue
            next_key = tuple(sorted(key + (file,)))
            mse_difference = value - file_mse[next_key]

            if file_of_interest in key:
                include_mse_effect.append((len(key), mse_difference))
            else:
                exclude_mse_effect.append((len(key), mse_difference))

    # 3. Organize MSE effects by number of files in the combination
    include_mse_by_num_files = {i: [] for i in range(1, len(file_names))}
    exclude_mse_by_num_files = {i: [] for i in range(1, len(file_names))}

    for num_files, mse_diff in include_mse_effect:
        include_mse_by_num_files[num_files].append(mse_diff)

    for num_files, mse_diff in exclude_mse_effect:
        exclude_mse_by_num_files[num_files].append(mse_diff)

    # 4. Plotting the results
    num_files_max = len(file_names) - 1  # Max number of files to consider
    end_at = end_at or num_files_max  # Use num_files_max if end_at is None

    # Create two sets of subplots: one for including the file, one for excluding
    fig, axes = plt.subplots(end_at - start_from + 1, 2, figsize=(14, (end_at - start_from + 1) * 4), constrained_layout=True)

    for i in range(start_from, end_at + 1):
        # MSE differences for combinations including and excluding the file_of_interest
        include_diffs = include_mse_by_num_files[i]
        exclude_diffs = exclude_mse_by_num_files[i]

        # T-test to check for significant differences
        if len(include_diffs) > 1 and len(exclude_diffs) > 1:
            t_stat, p_value = ttest_ind(include_diffs, exclude_diffs, equal_var=False)
        else:
            t_stat, p_value = None, None  # Not enough data for t-test

        # Print t-test results
        if p_value is not None:
            print(f'{i} Files: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f} {'(Significant)' if p_value < 0.05 else ''}')
        else:
            print(f'{i} Files: Not enough data for t-test.')

        # Plotting the distributions
        # Plot for combinations including the file
        sns.kdeplot(include_diffs, ax=axes[i - start_from, 0], color='blue', fill=True, warn_singular=False)
        axes[i - start_from, 0].set_title(f'{i} Files (Including "{file_of_interest}")')
        axes[i - start_from, 0].set_xlabel('MSE Difference')
        axes[i - start_from, 0].set_ylabel('Density')

        # Plot for combinations excluding the file
        sns.kdeplot(exclude_diffs, ax=axes[i - start_from, 1], color='red', fill=True, warn_singular=False)
        axes[i - start_from, 1].set_title(f'{i} Files (Excluding "{file_of_interest}")')
        axes[i - start_from, 1].set_xlabel('MSE Difference')
        axes[i - start_from, 1].set_ylabel('Density')

    plt.tight_layout()
    plt.show()

def plot_mse_distribution_with_ttest(file_mse, file_names, file_of_interest, start_from=1, end_at=None):
    """
    Plots the distribution of MSE values for combinations that include
    and exclude a given file of interest. Also performs a t-test for significance.

    Parameters:
    - file_mse: dictionary where keys are tuples of file names and values are MSEs
    - file_names: list of all file names
    - file_of_interest: the name of the file to analyze (e.g., 'Urtecho et al')
    - start_from: minimum number of files in the combination to plot
    - end_at: maximum number of files in the combination to plot
    """
    
    # Initialize lists to store MSE values
    include_mse_values = []
    exclude_mse_values = []
    
    # Calculate MSE values for combinations including and excluding the file_of_interest
    for key, value in file_mse.items():
        if file_of_interest in key:
            include_mse_values.append((len(key), value))
        else:
            exclude_mse_values.append((len(key), value))

    # Organize MSE values by number of files in the combination
    include_mse_by_num_files = {i: [] for i in range(1, len(file_names) + 1)}
    exclude_mse_by_num_files = {i: [] for i in range(1, len(file_names) + 1)}

    for num_files, mse_value in include_mse_values:
        include_mse_by_num_files[num_files].append(mse_value)

    for num_files, mse_value in exclude_mse_values:
        exclude_mse_by_num_files[num_files].append(mse_value)

    # Plotting the results
    num_files_max = len(file_names)-1
    end_at = end_at or num_files_max  # Use num_files_max if end_at is None

    # Create two sets of subplots: one for including the file, one for excluding
    fig, axes = plt.subplots(end_at - start_from + 1, 2, figsize=(14, (end_at - start_from + 1) * 4))

    for i in range(start_from, end_at + 1):
        # MSE values for combinations including and excluding the file_of_interest
        include_mse = include_mse_by_num_files[i]
        exclude_mse = exclude_mse_by_num_files[i]

        # T-test to check for significant differences
        if len(include_mse) > 1 and len(exclude_mse) > 1:
            t_stat, p_value = ttest_ind(include_mse, exclude_mse, equal_var=False)
        else:
            t_stat, p_value = None, None  # Not enough data for t-test

        # Print t-test results
        if p_value is not None:
            print(f'{i} Files: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f} {'(Significant)' if p_value < 0.05 else ''}')
        else:
            print(f'{i} Files: Not enough data for t-test.')

        # Plotting the distributions
        # Plot for combinations including the file
        sns.kdeplot(include_mse, ax=axes[i - start_from, 0], color='blue', fill=True, warn_singular=False)
        axes[i - start_from, 0].set_title(f'MSE Distribution for {i} Files (Including "{file_of_interest}")')
        axes[i - start_from, 0].set_xlabel('MSE Value')
        axes[i - start_from, 0].set_ylabel('Density')

        # Plot for combinations excluding the file
        sns.kdeplot(exclude_mse, ax=axes[i - start_from, 1], color='red', fill=True, warn_singular=False)
        axes[i - start_from, 1].set_title(f'MSE Distribution for {i} Files (Excluding "{file_of_interest}")')
        axes[i - start_from, 1].set_xlabel('MSE Value')
        axes[i - start_from, 1].set_ylabel('Density')

    plt.tight_layout()
    plt.show()
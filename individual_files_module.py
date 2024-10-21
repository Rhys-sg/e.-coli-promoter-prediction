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

import main_module

def evalue_each_file(file_names, split_data, preceding_message=''):

    loss_values = {name: [] for name in file_names}    

    # Create a model for each file
    for i, file in enumerate(file_names):
        print(f'{preceding_message}Training Model {i + 1}/{len(file_names)}', end='\r')

        # Train the model on the filtered data
        model, history, loss = main_module.run(split_data[file]['X_train'],
                                               split_data[file]['y_train'],
                                               split_data[file]['X_test'],
                                               split_data[file]['y_test'],
                                               file)
        loss_values[file].append(loss)

        # Train the model on all the data
        model, history, loss = main_module.run(split_data[file]['X_train'], 
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
    
    return all_results


def save_repeat_evalute_each_file(all_results, csv_path='Data/repeat_evalute_each_file.csv'):
    to_save = {}
    for key in all_results.keys():
        to_save[f'{key} (Self)'] = []
        to_save[f'{key} (All)'] = []

    for key, val in all_results.items():
        to_save[f'{key} (Self)'] = [each[0] for each in val]
        to_save[f'{key} (All)'] = [each[1] for each in val]

    pd.DataFrame(to_save).to_csv(csv_path)


def load_repeat_evalute_each_file(csv_path='Data/repeat_evalute_each_file.csv'):
    df = pd.read_csv(csv_path, index_col=0)
    temp = {}

    for col in df.columns:
        if ' (Self)' in col:
            key = col.replace(' (Self)', '')
            all_self = df[f'{key} (Self)'].tolist()
            all_all = df[f'{key} (All)'].tolist()
            
            temp[key] = list(zip(all_self, all_all))

    return temp


def save_repeat_evalute_each_file_statistics(all_results, split_data, file_names):
    data = {'File Name': [], 'Data Points' : [], 'Coefficient of Variation (Self)': [], 'Coefficient of Variation (All)': []}

    def calc_CV(data):
        return np.std(data) / np.mean(data)
    
    for file in file_names:
        data['File Name'].append(file)
        data['Data Points'].append(len(split_data[file]['X_train']))
        data['Coefficient of Variation (Self)'].append(calc_CV([each[0] for each in all_results[file]]))
        data['Coefficient of Variation (All)'].append(calc_CV([each[1] for each in all_results[file]]))
    
    pd.DataFrame(data).to_csv('Data/Figure 2.csv')


def plot_repeat_evalute_each_file(file_names, all_results):
    # Calculate the average and coefficient of variation for each file
    averaged_results = {file: np.mean(all_results[file], axis=0) for file in file_names}
    CV_training_data = {file: np.std([result[0] for result in all_results[file]]) / np.mean([result[0] for result in all_results[file]]) for file in file_names}
    CV_all = {file: np.std([result[1] for result in all_results[file]]) / np.mean([result[1] for result in all_results[file]]) for file in file_names}

    # Prepare the data
    sorted_file_names = sorted(file_names, key=lambda file: averaged_results[file][1])
    x = np.arange(len(sorted_file_names))
    bar_width = 0.35

    # Extracting the individual and average results
    training_data_mse = {file: [result[0] for result in all_results[file]] for file in sorted_file_names}
    all_data_mse = {file: [result[1] for result in all_results[file]] for file in sorted_file_names}
    avg_training_data = [averaged_results[file][0] for file in sorted_file_names]
    avg_all_data = [averaged_results[file][1] for file in sorted_file_names]

    plt.figure(figsize=(10, 6))

    for i, file in enumerate(sorted_file_names):
        plt.scatter([x[i] - bar_width / 2] * len(training_data_mse[file]), training_data_mse[file], color='skyblue', label='Training Data MSE' if i == 0 else "")
        plt.scatter([x[i] + bar_width / 2] * len(all_data_mse[file]), all_data_mse[file], color='lightgreen', label='All Data MSE' if i == 0 else "")
        
        # Add horizontal line for Training Data MSE average and All Data MSE average
        plt.hlines(avg_training_data[i], x[i] - bar_width / 2 - 0.05, x[i] - bar_width / 2 + 0.05, colors='grey', linestyles='solid', label='Avgerage MSE' if i == 0 else "")
        plt.hlines(avg_all_data[i], x[i] + bar_width / 2 - 0.05, x[i] + bar_width / 2 + 0.05, colors='grey', linestyles='solid', label='')
        
        # Add label for the coefficient of variation next to the horizontal lines
        plt.text(x[i] - bar_width*1.1, avg_training_data[i], f'{int(CV_training_data[file]*100)}%', ha='center', va='center', color='black', fontsize=8)
        plt.text(x[i] + bar_width*1.1, avg_all_data[i], f'{int(CV_all[file]*100)}%', ha='center', va='center', color='black', fontsize=8)

    # Formatting
    plt.xticks(x, sorted_file_names, rotation=45, ha='right', rotation_mode='anchor')
    plt.xlabel('Files')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Images/repeat_evalute_each_file.png')
    plt.savefig('Images/Figure 1.pdf', format='pdf')
    plt.show()
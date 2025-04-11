import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations, permutations

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
    
    pd.DataFrame(data).to_csv('Data/Table 1.csv')

def plot_repeat_evalute_each_file(file_names, all_results, file_order):
    # Calculate the average and coefficient of variation for each file
    averaged_results = {file: np.mean(all_results[file], axis=0) for file in file_names}
    averaged_training_data = {file: result[0] for file, result in averaged_results.items()}
    averaged_all_data = {file: result[1] for file, result in averaged_results.items()}
    CV_training_data = {file: np.std([result[0] for result in all_results[file]]) / np.mean([result[0] for result in all_results[file]]) for file in file_names}
    CV_all_data = {file: np.std([result[1] for result in all_results[file]]) / np.mean([result[1] for result in all_results[file]]) for file in file_names}
    
    # Determine sorted orders for training and all data
    training_data_order = sorted(file_names, key=lambda file: averaged_training_data[file])
    all_data_order = sorted(file_names, key=lambda file: averaged_all_data[file])
    
    # Ensure file order is respected
    file_order = [file for file in file_order if file in file_names]
    x = np.arange(len(file_order))
    bar_width = 0.35

    # Extracting the individual and average results
    training_data_mse = {file: [result[0] for result in all_results[file]] for file in file_order}
    all_data_mse = {file: [result[1] for result in all_results[file]] for file in file_order}
    avg_training_data = [averaged_training_data[file] for file in file_order]
    avg_all_data = [averaged_all_data[file] for file in file_order]

    # Plot for Training Data MSE
    plt.figure(figsize=(10, 6))
    for i, file in enumerate(file_order):
        plt.scatter([x[i]] * len(training_data_mse[file]), training_data_mse[file], color='skyblue', label='Training Data MSE' if i == 0 else "")
        plt.hlines(avg_training_data[i], x[i] - 0.1, x[i] + 0.1, colors='grey', linestyles='solid', label='Average MSE' if i == 0 else "")
    plt.ylim(0, max(max(max(training_data_mse.values())), max(avg_training_data)) * 1.2)
    for i, file in enumerate(file_order):
        plt.text(x[i], max(training_data_mse[file]) * 1.05, f'{int(CV_training_data[file]*100)}%', ha='center', va='bottom', color='black', fontsize=8)
    plt.xticks(x, file_order, rotation=45, ha='right', rotation_mode='anchor')
    plt.xlabel('Files')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training Data MSE')
    ##plt.legend()
    plt.tight_layout()
    plt.savefig('Images/training_data_mse.png')
    plt.savefig('Images/Figure_Training_Data.pdf', format='pdf')
    plt.show()

    # Plot for All Data MSE
    plt.figure(figsize=(10, 6))
    for i, file in enumerate(file_order):
        plt.scatter([x[i]] * len(all_data_mse[file]), all_data_mse[file], color='lightgreen', label='All Data MSE' if i == 0 else "")
        plt.hlines(avg_all_data[i], x[i] - 0.1, x[i] + 0.1, colors='grey', linestyles='solid', label='Average MSE' if i == 0 else "")
    plt.ylim(0, max(max(max(all_data_mse.values())), max(avg_all_data)) * 1.2)
    for i, file in enumerate(file_order):
        plt.text(x[i], max(all_data_mse[file]) * 1.05, f'{int(CV_all_data[file]*100)}%', ha='center', va='bottom', color='black', fontsize=8)
    plt.xticks(x, file_order, rotation=45, ha='right', rotation_mode='anchor')
    plt.xlabel('Files')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('All Data MSE')
    ##plt.legend()
    plt.tight_layout()
    plt.savefig('Images/all_data_mse.png')
    plt.savefig('Images/Figure_All_Data.pdf', format='pdf')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    for i, file in enumerate(file_order):
        ax1.scatter([x[i]] * len(training_data_mse[file]), training_data_mse[file], color='skyblue', label='Training Data MSE' if i == 0 else "")
        ax1.hlines(avg_training_data[i], x[i] - 0.1, x[i] + 0.1, colors='grey', linestyles='solid', label='Average MSE' if i == 0 else "")
    ax1.set_ylim(0, max(max(max(training_data_mse.values())), max(avg_training_data)) * 1.2)
    for i, file in enumerate(file_order):
        ax1.text(x[i], max(training_data_mse[file]) * 1.05, f'{int(CV_training_data[file]*100)}%', ha='center', va='bottom', color='black', fontsize=8)
    ax1.set_xticks(x, file_order, rotation=45, ha='right', rotation_mode='anchor')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.set_title('Testing against self')

    for i, file in enumerate(file_order):
        ax2.scatter([x[i]] * len(all_data_mse[file]), all_data_mse[file], color='lightgreen', label='All Data MSE' if i == 0 else "")
        ax2.hlines(avg_all_data[i], x[i] - 0.1, x[i] + 0.1, colors='grey', linestyles='solid', label='Average MSE' if i == 0 else "")
    ax2.set_ylim(0, max(max(max(all_data_mse.values())), max(avg_all_data)) * 1.2)
    for i, file in enumerate(file_order):
        ax2.text(x[i], max(all_data_mse[file]) * 1.05, f'{int(CV_all_data[file]*100)}%', ha='center', va='bottom', color='black', fontsize=8)
    ax2.set_xticks(x, file_order, rotation=45, ha='right', rotation_mode='anchor')
    ax2.set_title('Testing against aggregation')
    ##plt.savefig('Images/all_data_mse.png')
    ##plt.savefig('Images/Figure_All_Data.pdf', format='pdf')
    fig.supxlabel("Dataset")
    fig.tight_layout()
    fig.show()

    return all_data_order, training_data_order, averaged_training_data, averaged_all_data, CV_training_data, CV_all_data

# def plot_repeat_evalute_each_file(file_names, all_results):
#     # Calculate the average and coefficient of variation for each file
#     averaged_results = {file: np.mean(all_results[file], axis=0) for file in file_names}
#     averaged_training_data = {file: result[0] for file, result in averaged_results.items()}
#     averaged_all_data = {file: result[1] for file, result in averaged_results.items()}
#     CV_training_data = {file: np.std([result[0] for result in all_results[file]]) / np.mean([result[0] for result in all_results[file]]) for file in file_names}
#     CV_all_data = {file: np.std([result[1] for result in all_results[file]]) / np.mean([result[1] for result in all_results[file]]) for file in file_names}

#     # Prepare the data
#     training_data_order = sorted(file_names, key=lambda file: averaged_training_data[file])
#     all_data_order = sorted(file_names, key=lambda file: averaged_all_data[file])
#     x = np.arange(len(all_data_order))
#     bar_width = 0.35

#     # Extracting the individual and average results
#     training_data_mse = {file: [result[0] for result in all_results[file]] for file in all_data_order}
#     all_data_mse = {file: [result[1] for result in all_results[file]] for file in all_data_order}
#     avg_training_data = [averaged_training_data[file] for file in all_data_order]
#     avg_all_data = [averaged_all_data[file] for file in all_data_order]

#     plt.figure(figsize=(10, 6))

#     for i, file in enumerate(all_data_order):
#         plt.scatter([x[i] - bar_width / 2] * len(training_data_mse[file]), training_data_mse[file], color='skyblue', label='Training Data MSE' if i == 0 else "")
#         plt.scatter([x[i] + bar_width / 2] * len(all_data_mse[file]), all_data_mse[file], color='lightgreen', label='All Data MSE' if i == 0 else "")
        
#         # Add horizontal line for Training Data MSE average and All Data MSE average
#         plt.hlines(avg_training_data[i], x[i] - bar_width / 2 - 0.05, x[i] - bar_width / 2 + 0.05, colors='grey', linestyles='solid', label='Avgerage MSE' if i == 0 else "")
#         plt.hlines(avg_all_data[i], x[i] + bar_width / 2 - 0.05, x[i] + bar_width / 2 + 0.05, colors='grey', linestyles='solid', label='')
        
#         # Add label for the coefficient of variation next to the horizontal lines
#         plt.text(x[i] - bar_width*1.1, avg_training_data[i], f'{int(CV_training_data[file]*100)}%', ha='center', va='center', color='black', fontsize=8)
#         plt.text(x[i] + bar_width*1.1, avg_all_data[i], f'{int(CV_all_data[file]*100)}%', ha='center', va='center', color='black', fontsize=8)

#     # Formatting
#     plt.xticks(x, all_data_order, rotation=45, ha='right', rotation_mode='anchor')
#     plt.xlabel('Files')
#     plt.ylabel('Mean Squared Error (MSE)')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'Images/repeat_evalute_each_file.png')
#     plt.savefig('Images/Figure 1.pdf', format='pdf')
#     plt.show()

#     return all_data_order, training_data_order, averaged_training_data, averaged_all_data, CV_training_data, CV_all_data

'''
The next set of functions are used for the file comparison and analysis
'''

def add_to_df(df, dictionary):
    return [dictionary[file] for file in df['File Name']]

'''
Plotting functions for file comparison
'''

def _check_labels(y_axis, y_label, x_axis, x_label, title):
    if y_label is None:
        y_label = y_axis
    if x_label is None:
        x_label = x_axis
    if title is None:
        title = f'{y_label} vs {x_label}'
    return y_label, x_label, title

def plot_barchart(df, y_axis, y_label=None, x_axis='File Name', x_label=None, title=None, color='skyblue'):
    y_label, x_label, title = _check_labels(y_axis, y_label, x_axis, x_label, title)
    plt.figure(figsize=(10, 6))
    plt.bar(df[x_axis], df[y_axis], color=color, width=0.35)
    if any(value < 0 for value in df[y_axis]):
        plt.gca().invert_yaxis()
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def _dynamic_text_labels(x_pos, y_pos, x_min, x_max, y_min, y_max):
    if x_pos < x_min + (x_max - x_min) * 0.05:
        ha = 'left'
    elif x_pos > x_max - (x_max - x_min) * 0.05:
        ha = 'right'
    else:
        ha = 'center'
    if y_pos < y_min + (y_max - y_min) * 0.05:
        va = 'bottom'
    elif y_pos > y_max - (y_max - y_min) * 0.05:
        va = 'top'
    else:
        va = 'center'
    return ha, va

def _adjust_text_positions(texts, offsets, x_min, x_max, y_min, y_max):
    for i, (text, offset) in enumerate(zip(texts, offsets)):
        while any(
            text.get_window_extent().overlaps(other.get_window_extent())
            for j, other in enumerate(texts)
            if i != j
        ):
            # Find the x-coordinates of the current and other texts
            text_x = text.get_position()[0]
            other_xs = [
                other.get_position()[0]
                for j, other in enumerate(texts)
                if i != j and text.get_window_extent().overlaps(other.get_window_extent())
            ]

            # Adjust position to resolve overlap
            if text_x < min(other_xs, default=x_max):
                offset[0] -= (x_max - x_min) * 0.005  # Move left
            elif text_x > max(other_xs, default=x_min):
                offset[0] += (x_max - x_min) * 0.005  # Move right
            text.set_position((offset[0], text.get_position()[1]))


def plot_scatterplot(df, y_axis, x_axis, grouping='File Names', y_label=None, x_label=None, title=None, color='skyblue'):
    y_label, x_label, title = _check_labels(y_axis, y_label, x_axis, x_label, title)
    plt.figure(figsize=(10, 6))
    
    if 'Training' in y_axis or 'Training' in x_axis:
        color = 'skyblue'
    elif 'All' in y_axis or 'All' in x_axis:
        color = 'lightgreen'

    # Plot the scatter before adding the labels
    for i, row in df.iterrows():
        plt.scatter(row[x_axis], row[y_axis], color=color)

    # Find the limits of the plot
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    y_offset = (y_max - y_min) * 0.01

    # Add the labels to the scatter plot
    texts = []
    offsets = []
    for i, row in df.iterrows():
        x_pos = row[x_axis]
        y_pos = row[y_axis] + y_offset
        label = str(row[grouping])
        ha, _ = _dynamic_text_labels(x_pos, y_pos, x_min, x_max, y_min, y_max)
        if any(value < 0 for value in df[y_axis]):
            va='top'
        else:
            va='bottom'
        text = plt.text(x_pos, y_pos, label, fontsize=9, ha=ha, va=va)
        texts.append(text)
        offsets.append([x_pos, y_pos])

    # Adjust text positions to avoid overlap
    _adjust_text_positions(texts, offsets, x_min, x_max, y_min, y_max)

    if any(value < 0 for value in df[y_axis]):
        plt.gca().invert_yaxis()
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, y_axis, y_label=None, x_axis='File Name', x_label=None,  title=None, color='skyblue'):
    y_label, x_label, title = _check_labels(y_axis, y_label, x_axis, x_label, title)
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[y_axis], labels=df[x_axis], patch_artist=True, boxprops=dict(facecolor=color))
    if any(any(value < 0 for value in values) for values in df[y_axis]):
        plt.gca().invert_yaxis()
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_heatmap(df, y_label='File Name', x_label='File Name', title=None, flip_y=True, staircase=False, cmap='coolwarm'):
    if title is None:
        title = f'Heatmap of {y_label} vs {x_label}'
    mask = None
    if staircase:
        mask = np.triu(np.ones(df.shape), k=1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap=cmap, annot=True, fmt=".2f", mask=mask)
    if flip_y:
        plt.gca().invert_yaxis()
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def multiple_bar_chart(data, order, y_label, title, colors=None, normalize=True):
    total_width = 0.35
    num_plots = len(data)
    bar_width = total_width / num_plots

    if colors is None:
        colors = {label: plt.cm.tab20(i) for i, label in enumerate(data.keys())}

    # Normalize the data if required
    if normalize:
        data = {
            label: {key: abs(val / sum(inner_dict.values())) for key, val in inner_dict.items()}
            for label, inner_dict in data.items()
        }

    x = np.arange(len(order))
    plt.figure(figsize=(10, 6))
    for i, (label, inner_dict) in enumerate(data.items()):
        values = [inner_dict.get(file, 0) for file in order]
        plt.bar(x + (i - (num_plots - 1) / 2) * bar_width, values, bar_width, label=label, color=colors[label])
    
    plt.title(title, fontsize=16)
    plt.xlabel('File Name', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(rotation=45, ha='right', ticks=x, labels=order)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


'''
Data Analysis for file comparison
'''
def get_file_sequence_counts(df):
    return df['File Name'].value_counts().to_dict()

def get_promoter_sequence_lengths(df):
    return df.groupby('File Name')['Promoter Sequence'].apply(lambda x: x.str.len().tolist()).to_dict()

def get_observed_expressions(df):
    return df.groupby('File Name')['Observed log(TX/Txref)'].apply(lambda x: x.tolist()).to_dict()

def get_inter_file_entropy(df, normalize=True):
    return df.groupby('File Name').apply(lambda x: _calculate_average_entropy(x, normalize)).to_dict()

def get_pairwise_file_distance(df, n=10, order=None, pad=True, exclude_self=False, function=None):
    if order is not None:
        df = df[df['File Name'].isin(order)]
        file_names = order
    else:
        file_names = df['File Name'].unique()
    heatmap_data = np.zeros((len(file_names), len(file_names)))

    for i, file1 in enumerate(file_names):
        seqs_file1 = df[df['File Name'] == file1]['Promoter Sequence'].sample(n=min(n, len(df[df['File Name'] == file1]))).tolist()
        for j, file2 in enumerate(file_names):
            if i > j:
                continue
            if exclude_self and i == j:
                continue
            seqs_file2 = df[df['File Name'] == file2]['Promoter Sequence'].sample(n=min(n, len(df[df['File Name'] == file2]))).tolist()
            combined_seqs = seqs_file1 + seqs_file2
            avg_hamming = _average_pairwise_distance(combined_seqs, function)
            heatmap_data[i, j] = avg_hamming
            heatmap_data[j, i] = avg_hamming

    return pd.DataFrame(heatmap_data, index=file_names, columns=file_names)

def get_average_pairwise_distances(df, n=10, pad=True, exclude_self=False, function=None, normalize=False):
    file_names = df['File Name'].unique()
    pairwise_distances = {file: 0 for file in file_names}

    for i, file1 in enumerate(file_names):
        seqs_file1 = df[df['File Name'] == file1]['Promoter Sequence'].sample(n=min(n, len(df[df['File Name'] == file1]))).tolist()
        for j, file2 in enumerate(file_names):
            if i > j:
                continue
            if exclude_self and i == j:
                continue
            seqs_file2 = df[df['File Name'] == file2]['Promoter Sequence'].sample(n=min(n, len(df[df['File Name'] == file2]))).tolist()
            combined_seqs = seqs_file1 + seqs_file2
            pairwise_distances[file1] += _average_pairwise_distance(combined_seqs, function)
            pairwise_distances[file2] += _average_pairwise_distance(combined_seqs, function)

    for file in file_names:
        pairwise_distances[file] /= ((len(file_names) - 1) / 2)
    
    if normalize:
        max_distance = max(pairwise_distances.values())
        pairwise_distances = {file: pairwise_distances[file]/max_distance for file in file_names}

    return pairwise_distances
    

def plot_metric_mse(data, file_order, all_results, training_data=True, all_data=True):
    comiled_data = {}
    colors = {}

    if training_data:
        comiled_data['Training Data MSE'] = {file: np.mean(all_results[file], axis=0)[0] for file in file_order}
        colors['Training Data MSE'] = 'skyblue'
    if all_data:
        comiled_data['All Data MSE'] = {file: np.mean(all_results[file], axis=0)[1] for file in file_order}
        colors['All Data MSE'] = 'lightgreen'

    for metric, metric_data in data.items():
        comiled_data[metric] = metric_data
        colors[metric] = 'grey'
        multiple_bar_chart(comiled_data, file_order, y_label=f'', title=f'{metric} vs Model MSE', colors=colors)

        del comiled_data[metric]
        del colors[metric]

def plot_relative_data(data, file_order, all_results, training_data=True, all_data=True):
    comiled_data = {}
    colors = {}

    if training_data:
        comiled_data['Training Data MSE'] = {file: np.mean(all_results[file], axis=0)[0] for file in file_order}
        colors['Training Data MSE'] = 'skyblue'
    if all_data:
        comiled_data['All Data MSE'] = {file: np.mean(all_results[file], axis=0)[1] for file in file_order}
        colors['All Data MSE'] = 'lightgreen'

    for metric1, metric2 in permutations(data.keys(), 2):
        comiled_data[f'{metric1} / {metric2}'] = {file : data[metric1][file] / data[metric2][file] for file in file_order}
        colors[f'{metric1} / {metric2}'] = 'grey'
        multiple_bar_chart(comiled_data, file_order, y_label=f'', title='', colors=colors)

        del comiled_data[f'{metric1} / {metric2}']
        del colors[f'{metric1} / {metric2}']

def _calculate_inter_variance(df, normalize, pad=True):
    df['Promoter Sequence'] = df['Promoter Sequence'].apply(lambda x: x.upper())
    if pad or len(df['Promoter Sequence'].apply(lambda x: len(x)).unique()) > 1:
        df['Promoter Sequence'] = df['Promoter Sequence'].apply(lambda x: x.upper().zfill(150))

    variances = []
    for index in range(0, 150):
        frequency = {'A': 0, 'C': 0, 'G': 0, 'T': 0, '0': 0}
        for sequence in df['Promoter Sequence']:
            frequency[sequence[index]] += 1

        mean = sum(frequency.values()) / len(frequency)
        variance = sum([((x - mean) ** 2) for x in frequency.values()]) / len(frequency)
        total_count = sum(frequency.values())
        max_variance = ((total_count - mean) ** 2 + (len(frequency) - 1) * (0 - mean) ** 2) / len(frequency)

        variances.append(1 - (variance / max_variance))

    return sum(variances) / len(variances)

def _calculate_average_entropy(df, normalize):
    df['Promoter Sequence'] = df['Promoter Sequence'].apply(lambda x: x.upper())

    # Pad the sequences if necessary
    max_length = max(df['Promoter Sequence'].apply(lambda x: len(x)))
    if len(df['Promoter Sequence'].apply(lambda x: len(x)).unique()) > 1:
        df['Promoter Sequence'] = df['Promoter Sequence'].apply(lambda x: x.zfill(max_length))

    entropies = []
    for index in range(max_length):
        frequency = {'A': 0, 'C': 0, 'G': 0, 'T': 0, '0': 0}
        for sequence in df['Promoter Sequence']:
            frequency[sequence[index]] += 1
        total_count = sum(frequency.values())
        probabilities = [freq / total_count for freq in frequency.values() if freq > 0]
        entropy = -sum(p * np.log2(p) for p in probabilities)
        entropies.append(entropy)

    entropy = sum(entropies)
    if normalize:
        entropy /= max_length

    return entropy

def _average_pairwise_distance(sequences, function):
    distances = []
    for seq1, seq2 in combinations(sequences, 2):
        distances.append(function(seq1, seq2))
    return np.mean(distances) if distances else 0
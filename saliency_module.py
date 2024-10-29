import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from tensorflow.keras.models import load_model # type: ignore
import random
import math

import main_module

def plot_saliency_map_sequence(sequence, base_px=16, multiplier=1, model_filename='Models/model.keras'):
    
    model = load_model(model_filename)
    encoded_sequence, max_length = main_module.preprocess_sequences([sequence], 150)
    
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

def plot_saliency_map_grid(
    model_filename='Models/model.keras',
    data_filename='Data/LaFleur_supp.csv',
    data=None,
    i_start=0,
    i_end=None,
    relative=True,
    scaler=5,
    sort_by_prediction=True,
    title=None,
    ax=None
):
    model = load_model(model_filename)

    if data is None or data.empty:
        data = pd.read_csv(data_filename)

    if i_end is None:
        i_end = len(data)
    data = data.loc[i_start:i_end-1, 'Promoter Sequence']

    data, max_length = main_module.preprocess_sequences(data, 150)
    target_class_index = 0

    # Function to compute predictions and saliency maps
    def compute_prediction_and_saliency(sequence):
        prediction = model(tf.convert_to_tensor(sequence[np.newaxis, ...], dtype=tf.float32))[0, target_class_index]
        saliency = generate_saliency_map(model, sequence, target_class_index)
        if relative:
            saliency = saliency / (prediction * scaler)
        return prediction, saliency

    predictions_and_saliency = [compute_prediction_and_saliency(seq) for seq in data]

    # Optionally sort by prediction
    if sort_by_prediction:
        predictions_and_saliency.sort(key=lambda x: x[0])

    saliency_maps = [saliency for _, saliency in predictions_and_saliency]

    stacked_saliency_map = np.vstack(saliency_maps)
    vmin = np.min(stacked_saliency_map)
    vmax = np.max(stacked_saliency_map)

    # Plot on the provided subplot axis
    if not ax:
        plt.imshow(stacked_saliency_map, cmap='magma', aspect='auto', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        plt.show()
    else:
        im = ax.imshow(stacked_saliency_map, cmap='magma', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)

        return im


def plot_saliency_map_from_train_test_by_file(train_test_by_file):
    num_files = len(train_test_by_file)
    grid_size = math.ceil(math.sqrt(num_files))  # Ensure a square-like grid (3x3 if 9 files)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the 2D axes array to easily index them

    for idx, (file, train_test) in enumerate(train_test_by_file.items()):
        X_train, X_test, y_train, y_test = train_test
        X_test_data = main_module.decode_to_df(X_test)

        # Plot the saliency map on the corresponding subplot
        ax = axes[idx]
        plot_saliency_map_grid(data=X_test_data, i_end=100, title=file, ax=ax)

    # Hide any unused subplots (if the grid is larger than the number of files)
    for idx in range(len(train_test_by_file), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def generate_sequences(input_csv='Data/LaFleur.csv', output_csv='Data/Saliency/generated_sequences.csv', n=100, seed=None):
    if seed is not None:
        random.seed(seed)

    df = pd.read_csv(input_csv)
    columns = ['UP', 'h35', 'spacs', 'h10', 'disc', 'ITR']

    if not all(column in df.columns for column in columns):
        raise ValueError(f"Columns {columns} must be in the input CSV file")
    
    df = df[columns]
    sorted_entries = {column: df[column].value_counts().index.tolist() for column in df.columns}
    known_sequences = {(data['UP'], data['h35'], data['spacs'], data['h10'], data['disc'], data['ITR']) for _, data in df.iterrows()}

    # Generate X sequences not in known_sequences
    def _generate_sequences(df, sorted_entries, known_sequences, n, seed=seed):
        generated_sequences = set()
        
        while len(generated_sequences) < n:
            new_sequence = tuple(
                random.choice(sorted_entries[column][::-1]) for column in df.columns
            )
            
            if new_sequence not in known_sequences:
                generated_sequences.add(new_sequence)
            
        generated_sequences_df = pd.DataFrame(list(generated_sequences), columns=df.columns)
        generated_sequences_df['Promoter Sequence'] = pd.Series([''.join(sequence) for sequence in generated_sequences_df.values])

        return generated_sequences_df

    generated_sequences_df = _generate_sequences(df, sorted_entries, known_sequences, n, seed)
    generated_sequences_df.to_csv(output_csv, index=False)

    return generated_sequences_df

def generate_spacers(
    upstream_promoter='TTTTCTATCTACGTACTTGACA',
    downstream_promoter='TATAATAAACTTCCTCTACCTTAGTTTGTACGTT',
    output_csv='Data/Saliency/generated_spacers.csv',
    n=50,
    repeat_spacer='CTATTTCCTATTTCTCT',
    prefix_nucleotide=None,
):
    data = []
    i = 0

    if prefix_nucleotide is not None:
        while i < n:
            spacer = prefix_nucleotide * i
            sequence = upstream_promoter + spacer + downstream_promoter
            if len(sequence) >= 150:
                break
            data.append(sequence)
            i+=1
    else:
        spacer = ''
        while i < n:
            spacer += repeat_spacer[len(spacer) % len(repeat_spacer)]
            sequence = upstream_promoter + spacer + downstream_promoter
            if len(sequence) >= 150:
                break
            data.append(sequence)
            i += 1

    data_df = pd.DataFrame(data, columns=['Promoter Sequence'])
    data_df.to_csv(output_csv, index=False)

    return data_df
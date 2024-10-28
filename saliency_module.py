import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import random
from tensorflow.keras.models import load_model # type: ignore

import main_module

def plot_saliency_map_sequence(sequence, base_px=16, multiplier=1, model_filename='Models/model.keras'):
    """
    Visualizes the saliency map of the CNN model for a single sequence using a bar graph.

    Parameters:
     - model_filename (str): The filename of the model to load.
     - sequence (str): The sequence to visualize.
    """
    
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
    relative=False,
    scaler=1
):

    model = load_model(model_filename)

    if data is None or data.empty:
        data = pd.read_csv(data_filename)

    if i_end is None:
        i_end = len(data)
    data = data.loc[i_start:i_end-1, 'Promoter Sequence']

    data, max_length = main_module.preprocess_sequences(data, 150)
    target_class_index = 0

    # Function to compute scaled saliency maps if 'relative' is True
    def compute_scaled_saliency(sequence):
        saliency = generate_saliency_map(model, sequence, target_class_index)
        if relative:
            prediction = model(tf.convert_to_tensor(sequence[np.newaxis, ...], dtype=tf.float32))[0, target_class_index]
            saliency = saliency / (prediction * scaler)
        return saliency

    saliency_maps = [compute_scaled_saliency(sequence) for sequence in data]
    stacked_saliency_map = np.vstack(saliency_maps)
    vmin = np.min(stacked_saliency_map)
    vmax = np.max(stacked_saliency_map)

    plt.imshow(stacked_saliency_map, cmap='magma', aspect='auto', vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def generate_sequences(input_csv='Data/LaFleur.csv', output_csv='Data/generated_sequences.csv', X=100, seed=None):
    df = pd.read_csv(input_csv)
    columns = ['UP', 'h35', 'spacs', 'h10', 'disc', 'ITR']

    if not all(column in df.columns for column in columns):
        raise ValueError(f"Columns {columns} must be in the input CSV file")
    
    df = df[columns]
    sorted_entries = {column: df[column].value_counts().index.tolist() for column in df.columns}
    known_sequences = {(data['UP'], data['h35'], data['spacs'], data['h10'], data['disc'], data['ITR']) for _, data in df.iterrows()}

    # Generate X sequences not in known_sequences
    def generate_sequences(df, sorted_entries, known_sequences, X, seed=seed):
        generated_sequences = set()
        
        # Set the random seed if provided
        if seed is not None:
            random.seed(seed)

        while len(generated_sequences) < X:
            new_sequence = tuple(
                random.choice(sorted_entries[column][::-1]) for column in df.columns
            )
            
            if new_sequence not in known_sequences:
                generated_sequences.add(new_sequence)
            
        generated_sequences_df = pd.DataFrame(list(generated_sequences), columns=df.columns)
        generated_sequences_df['Promoter Sequence'] = pd.Series([''.join(sequence) for sequence in generated_sequences_df.values])

        return generated_sequences_df

    # Generate 5 new sequences not in known_sequences
    generated_sequences_df = generate_sequences(df, sorted_entries, known_sequences, X, seed)
    generated_sequences_df.to_csv(output_csv, index=False)

    return generated_sequences_df
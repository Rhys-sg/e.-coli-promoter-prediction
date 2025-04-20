import numpy as np
import tensorflow as tf
from keras.models import load_model  # type: ignore

class OneHotMethodWrapper:
    '''Descriptor to allow a method to work as both instance and class methods.'''
    def __get__(self, instance, owner):
        if instance is None:
            return lambda sequence, input_length=150: self.one_hot_sequence(sequence, input_length)
        return lambda sequence: self.one_hot_sequence(sequence, instance.input_length)

    @staticmethod
    def one_hot_sequence(sequence, input_length):
        '''One-hot encodes each nucleotide in the sequence and pads to uniform length.'''
        mapping = {
            'A': (1, 0, 0, 0),
            'C': (0, 1, 0, 0),
            'G': (0, 0, 1, 0),
            'T': (0, 0, 0, 1),
            '0': (0, 0, 0, 0),
            'N': (0.25, 0.25, 0.25, 0.25)
        }
        return tuple(mapping[nucleotide.upper()] for nucleotide in sequence.zfill(input_length))

class CNN:
    '''
    A wrapper for a Keras model that predicts the value of a given sequence.
    '''

    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.input_length = self.model.input_shape[1]
        self.cache = {}

    def preprocess(self, sequences):
        return [self.one_hot_sequence(seq) for seq in sequences]

    def predict(self, sequences):
        tensor_sequences = tf.convert_to_tensor(
            [list(seq) for seq in sequences],
            dtype=tf.float32
        )
        predictions = self.model(tensor_sequences, training=False)
        return tf.reshape(predictions, [-1])

    one_hot_sequence = OneHotMethodWrapper()

    @staticmethod
    def reverse_one_hot_sequence(one_hot_sequence, pad=False):
        '''Decodes a one-hot encoded sequence into a string of nucleotides.'''
        mapping = {
            (1, 0, 0, 0): 'A',
            (0, 1, 0, 0): 'C',
            (0, 0, 1, 0): 'G',
            (0, 0, 0, 1): 'T',
            (0, 0, 0, 0): '0' if pad else '',
            (0.25, 0.25, 0.25, 0.25): 'N'
        }
        return ''.join([mapping[tuple(nucleotide)] for nucleotide in one_hot_sequence])
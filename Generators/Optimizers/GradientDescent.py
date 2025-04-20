import random
import numpy as np
import tensorflow as tf
import os
from ..CNN_gradient import CNN

class GradientDescent:
    '''
    Gradient-based optimization to update nucleotide distributions at masked positions.
    Uses continuous relaxation and backpropagation to optimize nucleotide identities.
    '''

    def __init__(self, cnn_model_path, masked_sequence, target_expression, 
                 max_iter=100, learning_rate=0.1, seed=None):
        if seed is not None:
            self._set_seed(seed)

        self.cnn = CNN(cnn_model_path)
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.prediction_history = []
        self.error_history = []
        self.sequence_history = []
        self.early_stop = False

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if np.allclose(element, 0.25, atol=1e-9)]

    @staticmethod
    def _decode_soft_sequence(one_hot_array):
        '''Convert softmax outputs to hard one-hot and decode them.'''
        one_hot_hard = np.eye(4)[np.argmax(one_hot_array, axis=-1)]
        return CNN.reverse_one_hot_sequence(one_hot_hard)

    def run(self):
        logits = tf.Variable(tf.random.uniform((len(self.mask_indices), 4), minval=-0.1, maxval=0.1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        best_sequence = None
        best_prediction = None
        best_error = float('inf')

        for _ in range(self.max_iter):
            with tf.GradientTape() as tape:
                probs = tf.nn.softmax(logits, axis=-1)

                # Replace masked positions with learned probabilities
                base_seq = tf.convert_to_tensor(self.masked_sequence, dtype=tf.float32)
                full_seq = tf.tensor_scatter_nd_update(
                    base_seq,
                    indices=tf.constant([[i] for i in self.mask_indices], dtype=tf.int32),
                    updates=probs
                )

                # Forward pass with TensorFlow graph intact
                pred = self.cnn.predict(tf.expand_dims(full_seq, 0))
                loss = tf.abs(pred[0] - self.target_expression)

            grads = tape.gradient(loss, [logits])
            optimizer.apply_gradients(zip(grads, [logits]))

            # Convert to NumPy after gradient step
            predicted_val = pred[0].numpy()
            error_val = loss.numpy()
            decoded_seq = self._decode_soft_sequence(full_seq.numpy())

            self.prediction_history.append(predicted_val)
            self.error_history.append(error_val)
            self.sequence_history.append(decoded_seq)

            if error_val < best_error:
                best_error = error_val
                best_prediction = predicted_val
                best_sequence = decoded_seq

            if best_error == 0:
                self.early_stop = True
                break

        return best_sequence, best_prediction, best_error

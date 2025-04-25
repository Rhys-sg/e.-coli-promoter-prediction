import random
import numpy as np
import tensorflow as tf
import os
from ..CNN import CNN

class GuidedLocalSearch:
    '''
    Guided Local Search algorithm for optimizing sequences with a CNN prediction model.
    Escapes local optima by penalizing frequently visited features in the solution space.
    '''

    def __init__(self, cnn_model_path, masked_sequence, target_expression,
                 max_iter=1000, early_stopping_patience=None, penalty_factor=0.1, seed=None):
        if seed is not None:
            self._set_seed(seed)

        self.cnn = CNN(cnn_model_path)
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression
        self.max_iter = max_iter
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.penalty_factor = penalty_factor
        self.nucleotides = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1]   # T
        ])

        self.prediction_history = []
        self.error_history = []
        self.sequence_history = []
        self.penalties = {}

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if np.allclose(element, 0.25, atol=1e-9)]

    def _mutate_sequences(self, sequence):
        ''' Generate all possible mutations for the masked indices '''
        original_sequence = np.array(sequence, copy=True)
        return np.array([
            (mutated := original_sequence.copy(), mutated.__setitem__(idx, nt))[0]
            for idx in self.mask_indices
            for nt in self.nucleotides
            if not np.allclose(original_sequence[idx], nt)
        ])

    def _evaluate_sequences(self, sequences):
        predictions = self.cnn.predict(sequences, use_cache=False)
        errors = np.abs(self.target_expression - predictions)
        return predictions, errors

    def _augmented_cost(self, error, sequence):
        penalty = 0
        for idx in self.mask_indices:
            feature = tuple(sequence[idx])
            penalty += self.penalties.get((idx, feature), 0)
        return error + self.penalty_factor * penalty

    def _update_penalties(self, sequence):
        max_util = -1
        max_feature = None

        for idx in self.mask_indices:
            feature = tuple(sequence[idx])
            penalty = self.penalties.get((idx, feature), 0)
            util = (1 + penalty) * 1
            if util > max_util:
                max_util = util
                max_feature = (idx, feature)

        if max_feature:
            self.penalties[max_feature] = self.penalties.get(max_feature, 0) + 1

    def run(self):
        current_sequence = np.array(self.masked_sequence, copy=True)
        
        # randomly initialize the sequence
        for idx in self.mask_indices:
            random_nt = random.choice(self.nucleotides)
            current_sequence[idx] = random_nt
            
        current_prediction, current_error = self._evaluate_sequences([current_sequence])

        best_sequence = current_sequence.copy()
        best_prediction = current_prediction[0]
        best_error = current_error[0]

        self.prediction_history = [best_prediction]
        self.error_history = [best_error]
        self.sequence_history = [self.cnn.reverse_one_hot_sequence(current_sequence)]

        for _ in range(self.max_iter):
            mutated_sequences = self._mutate_sequences(current_sequence)

            if len(mutated_sequences) == 0:
                break

            predictions, errors = self._evaluate_sequences(mutated_sequences)

            best_candidate = None
            best_candidate_aug_cost = float('inf')
            best_candidate_prediction = None
            best_candidate_error = None

            for seq, pred, err in zip(mutated_sequences, predictions, errors):
                aug_cost = self._augmented_cost(err, seq)
                if aug_cost < best_candidate_aug_cost:
                    best_candidate = seq
                    best_candidate_prediction = pred
                    best_candidate_error = err
                    best_candidate_aug_cost = aug_cost

            if best_candidate is None:
                break

            current_sequence = best_candidate
            current_prediction = best_candidate_prediction
            current_error = best_candidate_error

            self.prediction_history.append(current_prediction)
            self.error_history.append(current_error)
            seq_key = self.cnn.reverse_one_hot_sequence(current_sequence)
            self.sequence_history.append(seq_key)

            if current_error < best_error:
                best_sequence = current_sequence.copy()
                best_prediction = current_prediction
                best_error = current_error
                self.early_stopping_counter = 0
            else:
                self._update_penalties(current_sequence)
                self.early_stopping_counter += 1

            if best_error == 0:
                break

            if self.early_stopping_patience != None and self.early_stopping_counter >= self.early_stopping_patience:
                break

        final_seq_str = self.cnn.reverse_one_hot_sequence(best_sequence)
        return final_seq_str, best_prediction, best_error
import random
import numpy as np
import tensorflow as tf
import os
from ..CNN import CNN

class GuidedLocalSearch:
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

    def _mutate_sequences(self, sequence, num_mutations=1):
        ''' Generate mutated sequences by randomly changing `num_mutations` masked positions '''
        mutated_sequences = []
        for _ in range(len(self.mask_indices) * 3):  # diversify mutations
            mutated = np.array(sequence, copy=True)
            indices = random.sample(self.mask_indices, min(num_mutations, len(self.mask_indices)))
            for idx in indices:
                new_nt = random.choice([nt for nt in self.nucleotides if not np.allclose(mutated[idx], nt)])
                mutated[idx] = new_nt
            mutated_sequences.append(mutated)
        return np.array(mutated_sequences)

    def _evaluate_sequences(self, sequences):
        predictions = self.cnn.predict(sequences, use_cache=False)
        errors = np.abs(self.target_expression - predictions)
        return predictions, errors

    def _augmented_cost(self, error, sequence):
        penalty = sum(self.penalties.get((idx, tuple(sequence[idx])), 0) for idx in self.mask_indices)
        return error + self.penalty_factor * penalty

    def _update_penalties(self, sequence, error):
        max_utility = -1
        max_feature = None

        for idx in self.mask_indices:
            feature = tuple(sequence[idx])
            penalty = self.penalties.get((idx, feature), 0)
            utility = error / (1 + penalty)
            if utility > max_utility:
                max_utility = utility
                max_feature = (idx, feature)

        if max_feature:
            self.penalties[max_feature] = self.penalties.get(max_feature, 0) + 1

    def run(self):
        current_sequence = np.array(self.masked_sequence, copy=True)

        for idx in self.mask_indices:
            current_sequence[idx] = random.choice(self.nucleotides)

        current_prediction, current_error = self._evaluate_sequences([current_sequence])
        best_sequence = current_sequence.copy()
        best_prediction = current_prediction[0]
        best_error = current_error[0]

        self.prediction_history = [best_prediction]
        self.error_history = [best_error]
        self.sequence_history = [self.cnn.reverse_one_hot_sequence(current_sequence)]

        for _ in range(self.max_iter):
            mutated_sequences = self._mutate_sequences(current_sequence, num_mutations=1)
            if len(mutated_sequences) == 0:
                break

            predictions, errors = self._evaluate_sequences(mutated_sequences)

            best_candidate = None
            best_aug_cost = float('inf')

            for seq, pred, err in zip(mutated_sequences, predictions, errors):
                aug_cost = self._augmented_cost(err, seq)
                if aug_cost < best_aug_cost:
                    best_candidate = seq
                    best_candidate_pred = pred
                    best_candidate_error = err
                    best_aug_cost = aug_cost

            if best_candidate is None:
                break

            current_sequence = best_candidate
            current_prediction = best_candidate_pred
            current_error = best_candidate_error

            self.prediction_history.append(current_prediction)
            self.error_history.append(current_error)
            self.sequence_history.append(self.cnn.reverse_one_hot_sequence(current_sequence))

            if current_error < best_error:
                best_sequence = current_sequence.copy()
                best_prediction = current_prediction
                best_error = current_error
                self.early_stopping_counter = 0
            else:
                self._update_penalties(current_sequence, current_error)
                self.early_stopping_counter += 1

            if best_error == 0:
                break

            if self.early_stopping_patience and self.early_stopping_counter >= self.early_stopping_patience:
                break

        return self.cnn.reverse_one_hot_sequence(best_sequence), best_prediction, best_error

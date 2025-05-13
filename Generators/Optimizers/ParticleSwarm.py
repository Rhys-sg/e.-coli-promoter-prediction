import random
import numpy as np
import tensorflow as tf
import os
from ..CNN import CNN
import pyswarms as ps

class ParticleSwarm:
    def __init__(self, cnn_model_path, masked_sequence, target_expression,
                 c1=0.43, c2=0.62, w=0.53, n_particles=20, max_iter=300,
                 early_stopping_patience=None, seed=None):
        if seed is not None:
            self._set_seed(seed)

        self.cnn = CNN(cnn_model_path)
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0

        self.nucleotides = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1]   # T
        ])

        self.prediction_history = []
        self.error_history = []
        self.infill_history = []
        self.n_bases = 4  # A, C, G, T

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if np.allclose(element, 0.25, atol=1e-9)]

    def _decode_particle(self, particle):
        sequence = np.array(self.masked_sequence, copy=True)
        for i, idx in enumerate(self.mask_indices):
            nucleotide_idx = int(np.clip(np.round(particle[i]), 0, 3))
            sequence[idx] = self.nucleotides[nucleotide_idx]
        return sequence

    def _evaluate_particles(self, swarm):
        errors = []
        for particle in swarm:
            sequence = self._decode_particle(particle)
            prediction = self.cnn.predict([sequence], use_cache=False)[0]
            error = abs(self.target_expression - prediction)
            errors.append(error)
        return np.array(errors)
    
    def run(self):
        dim = len(self.mask_indices)
        bounds = (np.zeros(dim), np.full(dim, 3))

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=dim,
            options={'c1': self.c1, 'c2': self.c2, 'w': self.w},
            bounds=bounds
        )

        best_error = float('inf')
        best_pos = None
        early_stopping_counter = 0

        for i in range(self.max_iter):
            cost, pos = optimizer.optimize(self._evaluate_particles, iters=1, verbose=False)
            current_best_sequence = self._decode_particle(pos)
            current_prediction = self.cnn.predict([current_best_sequence], use_cache=False)[0]
            current_error = abs(self.target_expression - current_prediction)

            self.prediction_history.append(current_prediction)
            self.error_history.append(current_error)
            self.infill_history.append(current_best_sequence)

            if current_error < best_error:
                best_error = current_error
                best_pos = pos
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if self.early_stopping_patience != None and early_stopping_counter >= self.early_stopping_patience:
                break

        best_sequence = self._decode_particle(best_pos)
        best_prediction = self.cnn.predict([best_sequence], use_cache=False)[0]
        best_error = abs(self.target_expression - best_prediction)

        return self.cnn.reverse_one_hot_sequence(best_sequence), best_prediction, best_error
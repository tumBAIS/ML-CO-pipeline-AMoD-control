import os
import tensorflow as tf
import numpy as np
import pandas as pd
from src import util
from pipeline.predictor import Predictor


class NNModel(Predictor):
    def __init__(self, args, rebalancing_grid=None, feature_data=None, clock=None, learning_rate=None):
        super().__init__(args, rebalancing_grid=rebalancing_grid, feature_data=feature_data, clock=clock)
        self.hidden_units = [self.args.hidden_units] * self.args.num_layers
        if args.mode in ["training"]:
            self.model = self.create_model(len(self.get_features()), len(self.get_features(without_relation=True))+3)
            self.model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.mean_squared_error])
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def save_model(self, args, step):
        file_output = util.get_learning_file(args, step, NNmodel=True)
        self.model.save(file_output)

    def load_model(self, parameters=None):
        if self.args.read_in_iterations == -1:  # load the last iteration
            self.args.read_in_iterations = self.get_last_trained_iteration()
        step = self.args.read_in_iterations if self.args.read_in_iterations is not None else 0
        self.model = tf.keras.models.load_model(util.get_learning_file(self.args, step, NNmodel=True))

    def get_last_trained_iteration(self):
        last_iteration = 0
        for learning_file in os.listdir(self.args.learned_parameters_directory):
            if (util.get_learning_file_name(self.args) in learning_file) and ("Learning_" in learning_file):
                iteration = int(learning_file.split("Learning_")[1].split("-")[0])
                if iteration > last_iteration:
                    last_iteration = iteration
        return last_iteration

    def forward_pass(self, data, perturbation=None):
        if len(data) == 0:
            return []
        else:
            weights = (-1 * np.squeeze(self.model(data))).tolist()
            if isinstance(weights, float):
                return [weights]
            else:
                return weights

    def create_model(self, input_dimension_edges, input_dimension_nodes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dimension_edges,)))
        if self.args.normalization:
            model.add(tf.keras.layers.Normalization(axis=1))
        for units in self.hidden_units:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(1, use_bias=False))
        return model

    def merge_features(self, training_data):
        return pd.concat([training_data["vehicles_requests"], training_data["requests_requests"],
                         training_data["vehicles_artRebVertices"], training_data["requests_artRebVertices"],
                         training_data["artRebVertices_artCapVertices"]]).values

    def grad_optimize(self, y, y_hat_mean, edge_features, node_features, edges):
        with tf.GradientTape() as tape_node:
            weights = tf.squeeze(self.model(edge_features, training=True))
            assert len(weights) == len(y) and len(weights) == len(y_hat_mean)
            correct_objective_profits = tf.math.reduce_sum(tf.multiply(weights, tf.squeeze(y.astype("float32"))))
            predicted_objective_profits = tf.math.reduce_sum(tf.multiply(weights, tf.squeeze(y_hat_mean.astype("float32"))))
            loss = tf.math.subtract(predicted_objective_profits, correct_objective_profits)
        self.optimizer.apply_gradients(zip(tape_node.gradient(loss, self.model.trainable_variables), self.model.trainable_variables))

"""
MLPRegressor model
"""
import os
import pandas
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor


class Model:
    name: str = None

    _base_path: str = os.path.dirname(os.path.realpath(__file__)) + '/dump/'
    _save_path: str = None

    _model: MLPRegressor = None

    def __init__(self, name, max_iter=200, verbose=2):
        self.name = name
        self._save_path = self._base_path + name + '.pickle'
        # Init or Load model from disk
        if os.path.exists(self._save_path):
            # Load the model from disk
            self._model = pickle.load(open(self._save_path, 'rb'))
        else:
            self._model = MLPRegressor(hidden_layer_sizes=[512, 256, 128, 64, 32], max_iter=max_iter, verbose=verbose)

    def train(self, X_train: pandas.DataFrame, y_train: pandas.DataFrame, print_error=True):
        self._model.fit(X_train, y_train)
        # Save the model on disk
        pickle.dump(self._model, open(self._save_path, 'wb'))
        if print_error:
            # Print the error
            p_train = self.predict(X_train)
            self.print_error(y_train, p_train)

    def predict(self, X: pandas.DataFrame) -> pandas.DataFrame:
        p = self._model.predict(X)
        return p

    @staticmethod
    def print_error(y: pandas.DataFrame, p: pandas.DataFrame):
        mae = mean_absolute_error(y, p)
        r2s = r2_score(y, p)
        print(f'Train: mean_absolute_error={mae}, r2_score={r2s}')

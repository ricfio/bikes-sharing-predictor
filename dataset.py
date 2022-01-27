"""
Generate custom data from master data
    hour.csv
        => hour_custom.csv

    Usage:
        ds = Dataset()
        ds.describe_master_data()
        ds.describe_custom_data()
        X_train, y_train = ds.get_train()
        X_test, y_test = ds.get_test()
"""
import os
import copy
import numpy as np
import pandas
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler


class Dataset:
    _base_path: str = 'data/'

    _master_path: str = _base_path + 'hour.csv'
    _custom_path: str = _base_path + 'hour_custom.csv'

    _master_data: pandas.DataFrame = None
    _custom_data: pandas.DataFrame = None
    _custom_data_for_train: pandas.DataFrame = None
    _custom_data_for_test: pandas.DataFrame = None

    _X_train: pandas.DataFrame = None
    _y_train: pandas.DataFrame = None

    _X_test: pandas.DataFrame = None
    _y_test: pandas.DataFrame = None

    def __init__(self):
        None

    # Replace a feature 'column' into a cyclical feature (splitting into 'column_sin', 'column_cos')
    # http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    @staticmethod
    def replace_cyclical_feature(data: pandas.DataFrame, column, value_counts, zero_based):
        if zero_based:
            data[column] = data[column] + 1
        data[column + '_sin'] = np.sin(2 * np.pi * data[column] / value_counts)
        data[column + '_cos'] = np.cos(2 * np.pi * data[column] / value_counts)
        data.drop(column, axis=1, inplace=True)

    # Describe a dataframe
    @staticmethod
    def _describe_data(data: pandas.DataFrame, name: str = None):
        print()
        if name is not None:
            print(f'DataFrame: {name}')
            print()
        df = data
        print(df.sample(10))
        print(df.columns)
        for column in df.columns:
            print()
            print(f'Feature: {column}')
            print(df[column].describe())

    # Return the master dataframe
    def get_master_data(self):
        if self._master_data is None:
            # Load the master data from csv
            self._master_data = pandas.read_csv(self._master_path)
        return self._master_data

    # Describe the master dataframe
    def describe_master_data(self):
        self.get_master_data()
        self._describe_data(self._master_data, 'master')

    # Return the custom dataframe
    def get_custom_data(self):
        if self._custom_data is None:
            if os.path.exists(self._custom_path):
                # Load the custom data from csv
                self._custom_data = pandas.read_csv(self._custom_path)
            else:
                # Make the custom data from master data
                self.get_master_data()
                self._custom_data = copy.deepcopy(self._master_data)
                # Replace the feature "hour" (hr = [0, 23]) into a cyclical feature
                self.replace_cyclical_feature(self._custom_data, 'hr', 24, True)
                # Replace the feature "month" (mnth = [1, 12]) into a cyclical feature
                self.replace_cyclical_feature(self._custom_data, 'mnth', 12, False)
                # Replace the feature "weekday" (weekday = [0,  6]) into a cyclical feature
                self.replace_cyclical_feature(self._custom_data, 'weekday', 7, True)
                # Drop the unwanted features
                columns_to_be_deleted = ['instant', 'dteday', 'casual', 'registered']
                self._custom_data.drop(columns_to_be_deleted, axis=1, inplace=True)
                # Transform some features
                transformers = [
                    ['one_hot', OneHotEncoder(), ['season', 'yr', 'weathersit']],
                    ['scaler', RobustScaler(), ['temp', 'atemp', 'hum', 'windspeed']]
                ]
                ct = ColumnTransformer(transformers, remainder='passthrough')
                ct.fit_transform(self._custom_data)
                # Save the custom data into csv
                self._custom_data.to_csv(self._custom_path, index=False)
            # Split the custom data into train & test
            np.random.seed(0)  # Init RNG (IMPORTANT to generate always the same train & test data)
            self._custom_data_for_train, self._custom_data_for_test = train_test_split(self._custom_data)
        return self._custom_data

    # Describe the custom dataframe
    def describe_custom_data(self):
        self.get_custom_data()
        self._describe_data(self._custom_data, 'custom')

    # Get train & test data
    def _get_train_test(self):
        if self._X_train is None or self._X_test is None or self._y_train is None or self._y_test is None:
            self.get_custom_data()
            y = self._custom_data['cnt']
            X = self._custom_data.drop('cnt', axis=1)
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y)
        return [self._X_train, self._X_test, self._y_train, self._y_test]

    # Get train data
    def get_train(self):
        self._get_train_test()
        return [self._X_train, self._y_train]

    # Get test data
    def get_test(self):
        self._get_train_test()
        return [self._X_test, self._y_test]

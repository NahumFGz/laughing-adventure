import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


class NumericalScalers:

    def __init__(self, dataset):
        self.dataset = dataset

    def scale(self, numerical_columns, method):
        """
        Esta función aplica el método de escalado especificado a las columnas numéricas.

        Args:
            numerical_columns (list): Lista de columnas numéricas.
            method (str): El método de escalado a aplicar. Debe ser uno de los siguientes:
                - 'StandardScaler'
                - 'MinMaxScaler'
                - 'MaxAbsScaler'
                - 'RobustScaler'
                - 'Normalizer'
                - 'PowerTransformer'

        Returns:
            DataFrame: El DataFrame con las columnas escaladas.

        Raises:
            ValueError: Si el método proporcionado no es uno de los esperados.
        """
        if method == "StandardScaler":
            return self.standard_scaler(numerical_columns)
        elif method == "MinMaxScaler":
            return self.min_max_scaler(numerical_columns)
        elif method == "MaxAbsScaler":
            return self.max_abs_scaler(numerical_columns)
        elif method == "RobustScaler":
            return self.robust_scaler(numerical_columns)
        elif method == "Normalizer":
            return self.normalizer(numerical_columns)
        elif method == "PowerTransformer":
            return self.power_transformer(numerical_columns)
        else:
            raise ValueError(
                f'Invalid method: {method}. Expected one of ["StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "Normalizer", "PowerTransformer"].'
            )

    def standard_scaler(self, numerical_columns):
        """
        Aplicar Standard Scaling a las columnas numéricas.

        Args:
            numerical_columns (list): Lista de columnas numéricas.

        Returns:
            DataFrame: El DataFrame con las columnas numéricas escaladas.
        """
        data = self.dataset.copy()
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        return data

    def min_max_scaler(self, numerical_columns):
        """
        Aplicar Min-Max Scaling a las columnas numéricas.

        Args:
            numerical_columns (list): Lista de columnas numéricas.

        Returns:
            DataFrame: El DataFrame con las columnas numéricas escaladas.
        """
        data = self.dataset.copy()
        scaler = MinMaxScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        return data

    def max_abs_scaler(self, numerical_columns):
        """
        Aplicar Max-Abs Scaling a las columnas numéricas.

        Args:
            numerical_columns (list): Lista de columnas numéricas.

        Returns:
            DataFrame: El DataFrame con las columnas numéricas escaladas.
        """
        data = self.dataset.copy()
        scaler = MaxAbsScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        return data

    def robust_scaler(self, numerical_columns):
        """
        Aplicar Robust Scaling a las columnas numéricas.

        Args:
            numerical_columns (list): Lista de columnas numéricas.

        Returns:
            DataFrame: El DataFrame con las columnas numéricas escaladas.
        """
        data = self.dataset.copy()
        scaler = RobustScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        return data

    def normalizer(self, numerical_columns):
        """
        Aplicar Normalizer a las columnas numéricas.

        Args:
            numerical_columns (list): Lista de columnas numéricas.

        Returns:
            DataFrame: El DataFrame con las columnas numéricas normalizadas.
        """
        data = self.dataset.copy()
        scaler = Normalizer()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        return data

    def power_transformer(self, numerical_columns):
        """
        Aplicar Power Transformer a las columnas numéricas.

        Args:
            numerical_columns (list): Lista de columnas numéricas.

        Returns:
            DataFrame: El DataFrame con las columnas numéricas transformadas.
        """
        data = self.dataset.copy()
        scaler = PowerTransformer()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        return data

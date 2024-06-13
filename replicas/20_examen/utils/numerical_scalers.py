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

    def provider(self, method):
        """
        Esta función aplica el método de escalado especificado a todas las columnas del dataset.

        Args:
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
            return self.standard_scaler()
        elif method == "MinMaxScaler":
            return self.min_max_scaler()
        elif method == "MaxAbsScaler":
            return self.max_abs_scaler()
        elif method == "RobustScaler":
            return self.robust_scaler()
        elif method == "Normalizer":
            return self.normalizer()
        elif method == "PowerTransformer":
            return self.power_transformer()
        else:
            raise ValueError(
                f'Invalid method: {method}. Expected one of ["StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "Normalizer", "PowerTransformer"].'
            )

    def standard_scaler(self):
        """
        Aplicar Standard Scaling a todas las columnas del dataset.

        Returns:
            DataFrame: El DataFrame con las columnas escaladas.
        """
        data = self.dataset.copy()
        scaler = StandardScaler()
        data[:] = scaler.fit_transform(data)
        return data

    def min_max_scaler(self):
        """
        Aplicar Min-Max Scaling a todas las columnas del dataset.

        Returns:
            DataFrame: El DataFrame con las columnas escaladas.
        """
        data = self.dataset.copy()
        scaler = MinMaxScaler()
        data[:] = scaler.fit_transform(data)
        return data

    def max_abs_scaler(self):
        """
        Aplicar Max-Abs Scaling a todas las columnas del dataset.

        Returns:
            DataFrame: El DataFrame con las columnas escaladas.
        """
        data = self.dataset.copy()
        scaler = MaxAbsScaler()
        data[:] = scaler.fit_transform(data)
        return data

    def robust_scaler(self):
        """
        Aplicar Robust Scaling a todas las columnas del dataset.

        Returns:
            DataFrame: El DataFrame con las columnas escaladas.
        """
        data = self.dataset.copy()
        scaler = RobustScaler()
        data[:] = scaler.fit_transform(data)
        return data

    def normalizer(self):
        """
        Aplicar Normalizer a todas las columnas del dataset.

        Returns:
            DataFrame: El DataFrame con las columnas normalizadas.
        """
        data = self.dataset.copy()
        scaler = Normalizer()
        data[:] = scaler.fit_transform(data)
        return data

    def power_transformer(self):
        """
        Aplicar Power Transformer a todas las columnas del dataset.

        Returns:
            DataFrame: El DataFrame con las columnas transformadas.
        """
        data = self.dataset.copy()
        scaler = PowerTransformer()
        data[:] = scaler.fit_transform(data)
        return data

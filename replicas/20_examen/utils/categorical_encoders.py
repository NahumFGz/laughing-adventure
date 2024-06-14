import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class AuxiliaryFunctions:
    def _map_binary_columns(self, data, binary_columns):
        """
        Mapear los valores binarios a 1 y 0.

        Args:
            data (DataFrame): El DataFrame de entrada.
            binary_columns (list): Lista de columnas binarias a mapear.

        Returns:
            DataFrame: El DataFrame con las columnas binarias mapeadas.
        """
        for column in binary_columns:
            unique_values = data[column].unique()
            if len(unique_values) == 2:
                data[column] = data[column].map({unique_values[0]: 1, unique_values[1]: 0})

        return data

    def get_binary_categorical_columns(self):
        """
        Identificar las columnas binarias y categóricas en el dataset.

        Returns:
            tuple: Dos listas, una con columnas binarias y otra con columnas categóricas.
        """
        binary_columns = []
        categorical_columns = []

        for column in self.dataset.columns:
            if self.dataset[column].dtype == "object":
                unique_values = self.dataset[column].nunique()
                if unique_values == 2:
                    binary_columns.append(column)
                else:
                    categorical_columns.append(column)

        return binary_columns, categorical_columns


class CategoricalEncoders(AuxiliaryFunctions):

    def __init__(self, dataset):
        self.dataset = dataset

    def provider(self, binary_columns, categorical_columns, method):
        """
        Esta función aplica el método de codificación especificado a las columnas binarias y categóricas.

        Args:
            binary_columns (list): Lista de columnas binarias.
            categorical_columns (list): Lista de columnas categóricas.
            method (str): El método de codificación a aplicar. Debe ser uno de los siguientes:
                - 'LabelEncoder'
                - 'OneHotEncoder'
                - 'OrdinalEncoder'
                - 'FrequencyEncoder'
                - 'BinaryEncoder'
                - 'BackwardDifferenceEncoder'

        Returns:
            DataFrame: El DataFrame con las columnas codificadas.

        Raises:
            ValueError: Si el método proporcionado no es uno de los esperados.
        """
        if method == "LabelEncoder":
            return self.label_encoder(binary_columns, categorical_columns)
        elif method == "OneHotEncoder":
            return self.one_hot_encoder(binary_columns, categorical_columns)
        elif method == "OrdinalEncoder":
            return self.ordinal_encoder(binary_columns, categorical_columns)
        elif method == "FrequencyEncoder":
            return self.frequency_encoder(binary_columns, categorical_columns)
        elif method == "BinaryEncoder":
            return self.binary_encoder(binary_columns, categorical_columns)
        elif method == "BackwardDifferenceEncoder":
            return self.backward_difference_encoder(binary_columns, categorical_columns)
        else:
            raise ValueError(
                f'Invalid method: {method}. Expected one of ["LabelEncoder", "OneHotEncoder", "OrdinalEncoder", "FrequencyEncoder", "BinaryEncoder", "BackwardDifferenceEncoder"].'
            )

        return self

    def label_encoder(self, binary_columns, categorical_columns):
        """
        Aplicar Label Encoding a las columnas categóricas no binarias.

        Args:
            binary_columns (list): Lista de columnas binarias.
            categorical_columns (list): Lista de columnas categóricas.

        Returns:
            DataFrame: El DataFrame con las columnas categóricas codificadas.
        """
        data = self.dataset.copy()

        # Mapear las columnas binarias a valores numéricos
        data = self._map_binary_columns(data, binary_columns)

        # Aplicar Label Encoding a las columnas categóricas
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        return data

    def one_hot_encoder(self, binary_columns, categorical_columns):
        """
        Aplicar One Hot Encoding a las columnas categóricas no binarias.

        Args:
            binary_columns (list): Lista de columnas binarias.
            categorical_columns (list): Lista de columnas categóricas.

        Returns:
            DataFrame: El DataFrame con las columnas categóricas codificadas.
        """
        data = self.dataset.copy()

        # Mapear las columnas binarias a valores numéricos
        data = self._map_binary_columns(data, binary_columns)

        # Aplicar One Hot Encoding a las columnas categóricas
        encoder = OneHotEncoder(drop="first", sparse_output=False)
        encoded_data = encoder.fit_transform(data[categorical_columns])

        # Obtener nombres de las columnas codificadas
        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)

        # Concatenar el DataFrame original con las columnas codificadas
        final_data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)

        return final_data

    def ordinal_encoder(self, binary_columns, categorical_columns):
        """
        Aplicar Ordinal Encoding a las columnas categóricas no binarias.

        Args:
            binary_columns (list): Lista de columnas binarias.
            categorical_columns (list): Lista de columnas categóricas.

        Returns:
            DataFrame: El DataFrame con las columnas categóricas codificadas.
        """
        data = self.dataset.copy()

        # Mapear las columnas binarias a valores numéricos
        data = self._map_binary_columns(data, binary_columns)

        # Aplicar Ordinal Encoding a las columnas categóricas
        encoder = ce.OrdinalEncoder(cols=categorical_columns)
        data_encoded = encoder.fit_transform(data)

        return data_encoded

    def frequency_encoder(self, binary_columns, categorical_columns):
        """
        Aplicar Frequency Encoding a las columnas categóricas.

        Args:
            binary_columns (list): Lista de columnas binarias.
            categorical_columns (list): Lista de columnas categóricas.

        Returns:
            DataFrame: El DataFrame con las columnas categóricas codificadas usando Frequency Encoding.
        """
        data = self.dataset.copy()
        data = self._map_binary_columns(data, binary_columns)

        # Aplicar Frequency Encoding a las columnas categóricas
        for col in categorical_columns:
            frequency = data[col].value_counts(normalize=True)
            data[col] = data[col].map(frequency)

        return data

    def binary_encoder(self, binary_columns, categorical_columns):
        """
        Aplicar Binary Encoding a las columnas categóricas.

        Args:
            binary_columns (list): Lista de columnas binarias.
            categorical_columns (list): Lista de columnas categóricas.

        Returns:
            DataFrame: El DataFrame con las columnas categóricas codificadas usando Binary Encoding.
        """
        data = self.dataset.copy()
        data = self._map_binary_columns(data, binary_columns)

        # Aplicar Binary Encoding a las columnas categóricas
        encoder = ce.BinaryEncoder(cols=categorical_columns)
        data_encoded = encoder.fit_transform(data)

        return data_encoded

    def backward_difference_encoder(self, binary_columns, categorical_columns):
        """
        Aplicar Backward Difference Encoding a las columnas categóricas no binarias.

        Args:
            binary_columns (list): Lista de columnas binarias.
            categorical_columns (list): Lista de columnas categóricas.

        Returns:
            DataFrame: El DataFrame con las columnas categóricas codificadas.
        """
        data = self.dataset.copy()
        data = self._map_binary_columns(data, binary_columns)
        encoder = ce.BackwardDifferenceEncoder(cols=categorical_columns)
        data_encoded = encoder.fit_transform(data)
        return data_encoded


class CategoricalEncodersExtra(AuxiliaryFunctions):

    def hashing_encoder(self, binary_columns, categorical_columns, n_components=8):
        """
        Aplicar Hashing Encoding a las columnas categóricas no binarias.

        Args:
            binary_columns (list): Lista de columnas binarias.
            categorical_columns (list): Lista de columnas categóricas.
            n_components (int): Número de componentes para el hashing.

        Returns:
            DataFrame: El DataFrame con las columnas categóricas codificadas.
        """
        data = self.dataset.copy()
        data = self._map_binary_columns(data, binary_columns)
        encoder = ce.HashingEncoder(cols=categorical_columns, n_components=n_components)
        data_encoded = encoder.fit_transform(data)
        return data_encoded

    def target_encoder(self, binary_columns, categorical_columns, target):
        """
        Aplicar Target Encoding a las columnas categóricas no binarias.

        Args:
            binary_columns (list): Lista de columnas binarias.
            categorical_columns (list): Lista de columnas categóricas.
            target (Series): Serie que contiene la variable objetivo.

        Returns:
            DataFrame: El DataFrame con las columnas categóricas codificadas usando Target Encoding.
        """
        data = self.dataset.copy()
        data = self._map_binary_columns(data, binary_columns)

        # Aplicar Target Encoding a las columnas categóricas
        encoder = ce.TargetEncoder(cols=categorical_columns)
        data_encoded = encoder.fit_transform(data[categorical_columns], target)
        data[categorical_columns] = data_encoded

        return data

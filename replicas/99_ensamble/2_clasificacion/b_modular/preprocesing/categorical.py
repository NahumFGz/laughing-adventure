import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Categorical:

    def __init__(self, dataset):
        self.dataset = dataset

    def _map_binary_columns(self, data, binary_columns):
        """
        Mapear los valores binarios a 1 y 0.

        Args:
            data (DataFrame): El DataFrame de entrada.
            binary_columns (list): Lista de columnas binarias a mapear.

        Returns:
            DataFrame: El DataFrame con las columnas binarias mapeadas.
        """
        binary_map = {"SI": 1, "NO": 0}
        data[binary_columns] = data[binary_columns].applymap(lambda x: binary_map.get(x, x))
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
        # Usa drop='first' para evitar multicolinealidad
        encoder = OneHotEncoder(drop="first")
        encoded_data = encoder.fit_transform(data[categorical_columns]).toarray()

        # Obtener nombres de las columnas codificadas
        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

        # Concatenar el DataFrame original con las columnas codificadas
        final_data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)

        return final_data

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

    def categorical_encoder(self, binary_columns, categorical_columns):
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

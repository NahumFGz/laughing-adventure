import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Categorical:

    def __init__(self, dataset):
        self.dataset = dataset

    def _map_binary_columns(self, data, binary_columns):
        # Mapea los valores binarios "SI" y "NO" a 1 y 0 respectivamente
        binary_map = {"SI": 1, "NO": 0}
        data[binary_columns] = data[binary_columns].applymap(lambda x: binary_map.get(x, x))
        return data

    def get_binary_categorical_columns(self):
        # Identifica y separa las columnas binarias y categóricas en el dataset
        binary_columns = []
        categorical_columns = []

        for column in self.dataset.columns:
            # Asume que todas las categóricas son de tipo 'object'
            if self.dataset[column].dtype == "object":
                unique_values = self.dataset[column].nunique()
                # Si la columna tiene solo 2 valores únicos, es binaria
                if unique_values == 2:
                    binary_columns.append(column)
                else:
                    categorical_columns.append(column)

        return binary_columns, categorical_columns

    def one_hot_encoder(self, binary_columns, categorical_columns):
        # Crea una copia del dataset original
        data = self.dataset.copy()

        # Mapea las columnas binarias a valores numéricos
        data = self._map_binary_columns(data, binary_columns)

        # Prepara el OneHotEncoder para variables categóricas no binarias
        encoder = OneHotEncoder(drop="first")  # Usa drop='first' para evitar multicolinealidad
        encoded_data = encoder.fit_transform(data[categorical_columns]).toarray()

        # Crear nombres de columnas para los datos codificados
        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

        # Concatenar el DataFrame original con el DataFrame de variables codificadas
        final_data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)

        return final_data

    def label_encoder(self, binary_columns, categorical_columns):
        # Crea una copia del dataset original
        data = self.dataset.copy()

        # Mapea las columnas binarias a valores numéricos
        data = self._map_binary_columns(data, binary_columns)

        # Aplica Label Encoding a las columnas categóricas no binarias
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le  # Guarda el encoder para posible uso futuro

        return data

    def categorical_encoder(self, binary_columns, categorical_columns):
        # Crea una copia del dataset original
        data = self.dataset.copy()

        # Mapea las columnas binarias a valores numéricos
        data = self._map_binary_columns(data, binary_columns)

        # Aplica Ordinal Encoding a las columnas categóricas no binarias
        encoder = ce.OrdinalEncoder(cols=categorical_columns)
        data_encoded = encoder.fit_transform(data)

        return data_encoded

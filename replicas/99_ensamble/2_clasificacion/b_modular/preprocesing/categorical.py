import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Categorical:

    def __init__(self, dataset):
        self.dataset = dataset

    def get_binary_categorical_columns(self, dataset):
        # Identificar las columnas binarias y categóricas
        binary_columns = []
        categorical_columns = []

        for column in dataset.columns:
            # Asumiendo que todas las categóricas son tipo 'object'
            if dataset[column].dtype == "object":
                unique_values = dataset[column].nunique()
                if unique_values == 2:
                    binary_columns.append(column)
                else:
                    categorical_columns.append(column)

        binary_columns, categorical_columns

    def one_hot_encoder(self, binary_columns):
        encoder = OneHotEncoder()
        # Eliminar la columna 'Codigo'

        data = self.dataset.copy()
        data.drop("Codigo", axis=1, inplace=True)

        # Identificar las columnas binarias y mapear los valores a numéricos
        binary_map = {"SI": 1, "NO": 0}
        data[binary_columns] = data[binary_columns].applymap(lambda x: binary_map.get(x, x))

        # Preparar el encoder para variables categóricas no binarias
        categorical_columns = data.select_dtypes(include=["object"]).columns.difference(
            binary_columns
        )
        encoder = OneHotEncoder(drop="first")  # Usar drop='first' para evitar multicolinealidad
        encoded_data = encoder.fit_transform(data[categorical_columns]).toarray()

        # Crear un DataFrame con las columnas codificadas y asignar nombres apropiados a las columnas
        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

        # Concatenar el DataFrame original con el nuevo DataFrame de variables codificadas
        final_data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)

        return final_data

    def label_encoder(self, binary_columns):
        data = self.dataset.copy()

        # Eliminar la columna 'Codigo'
        data.drop("Codigo", axis=1, inplace=True)

        # Identificar las columnas binarias y mapear los valores a numéricos
        binary_map = {"SI": 1, "NO": 0}
        data[binary_columns] = data[binary_columns].applymap(lambda x: binary_map.get(x, x))

        # Identificar las columnas categóricas que no son binarias
        categorical_columns = data.select_dtypes(include=["object"]).columns.difference(
            binary_columns
        )

        # Aplicar Label Encoding a las columnas categóricas
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = (
                le  # Guardar el encoder por si necesitas invertir el encoding más tarde
            )

        return data

    def categorical_encoder(self, binary_columns):
        # Cargar los datos
        data = self.dataset.copy()

        # Eliminar la columna 'Codigo'
        data.drop("Codigo", axis=1, inplace=True)

        # Identificar las columnas binarias y mapear los valores a numéricos
        binary_map = {"SI": 1, "NO": 0}
        data[binary_columns] = data[binary_columns].applymap(lambda x: binary_map.get(x, x))

        # Identificar las columnas categóricas que no son binarias
        categorical_columns = data.select_dtypes(include=["object"]).columns.difference(
            binary_columns
        )

        # Crear el encoder
        encoder = ce.OrdinalEncoder(cols=categorical_columns)

        # Aplicar el encoder al DataFrame
        data_encoded = encoder.fit_transform(data)

        # Ver las primeras filas del DataFrame procesado
        data_encoded.head()

        return data_encoded

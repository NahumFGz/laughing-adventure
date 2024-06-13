from catboost import CatBoostClassifier
from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier
from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
from cuml.naive_bayes import GaussianNB as cuMLGaussianNB
from cuml.neighbors import KNeighborsClassifier as cuMLKNeighborsClassifier
from cuml.svm import SVC as cuMLSVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class BaseModels:
    """
    Clase para seleccionar diferentes modelos de clasificación.

    Métodos:
    - 'logistic_regression': Regresión Logística.
    - 'decision_tree': Árbol de Decisión.
    - 'random_forest': Bosque Aleatorio.
    - 'svm': Support Vector Machine.
    - 'knn': K-Nearest Neighbors.
    - 'naive_bayes': Naive Bayes.
    - 'lgbm': LightGBM Classifier.
    - 'catboost': CatBoost Classifier.
    - 'xgboost': XGBoost Classifier.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def provider(self, method):
        """
        Devuelve un modelo de clasificación basado en el método especificado.

        Args:
            method (str): Método de clasificación.

        Returns:
            object: Modelo de clasificación seleccionado.

        Raises:
            ValueError: Si el método no es uno de los esperados.
        """
        if method == "logistic_regression":
            return cuMLLogisticRegression(random_state=self.random_state)
        elif method == "random_forest":
            return cuMLRandomForestClassifier(random_state=self.random_state)
        elif method == "svm":
            return cuMLSVC(probability=True, random_state=self.random_state)
        elif method == "knn":
            return cuMLKNeighborsClassifier()
        elif method == "naive_bayes":
            return cuMLGaussianNB()
        elif method == "lgbm":
            return LGBMClassifier(random_state=self.random_state)
        elif method == "catboost":
            return CatBoostClassifier(random_state=self.random_state, verbose=0)
        elif method == "xgboost":
            return XGBClassifier(random_state=self.random_state)
        else:
            raise ValueError("Invalid classification method specified.")

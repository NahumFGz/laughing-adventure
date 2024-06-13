from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class BaseModels:
    """
    Clase para seleccionar diferentes modelos de clasificación.

    Métodos:
    - 'logistic_regression': Regresión Logística.
    - 'decision_tree': Árbol de Decisión.
    - 'random_forest': Bosque Aleatorio.
    - 'gradient_boosting': Gradient Boosting.
    - 'svm': Support Vector Machine.
    - 'knn': K-Nearest Neighbors.
    - 'naive_bayes': Naive Bayes.
    - 'mlp': Perceptrón Multicapa (Red Neuronal).
    - 'lgbm': LightGBM Classifier.
    - 'catboost': CatBoost Classifier.
    - 'xgboost': XGBoost Classifier.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def provider(self, method):
        """
        Esta función devuelve un modelo de clasificación basado en el método especificado.

        Args:
            method (str): El método de clasificación a utilizar. Debe ser uno de los siguientes:
                - 'logistic_regression'
                - 'decision_tree'
                - 'random_forest'
                - 'gradient_boosting'
                - 'svm'
                - 'knn'
                - 'naive_bayes'
                - 'mlp'
                - 'lgbm'
                - 'catboost'
                - 'xgboost'

        Returns:
            object: El modelo de clasificación seleccionado.

        Raises:
            ValueError: Si el método proporcionado no es uno de los esperados.
        """
        if method == "logistic_regression":
            return LogisticRegression(random_state=self.random_state)
        elif method == "decision_tree":
            return DecisionTreeClassifier(random_state=self.random_state)
        elif method == "random_forest":
            return RandomForestClassifier(random_state=self.random_state)
        elif method == "gradient_boosting":
            return GradientBoostingClassifier(random_state=self.random_state)
        elif method == "svm":
            return SVC(probability=True, random_state=self.random_state)
        elif method == "knn":
            return KNeighborsClassifier()
        elif method == "naive_bayes":
            return GaussianNB()
        elif method == "mlp":
            return MLPClassifier(random_state=self.random_state)
        elif method == "lgbm":
            return LGBMClassifier(random_state=self.random_state)
        elif method == "catboost":
            return CatBoostClassifier(random_state=self.random_state, verbose=0)
        elif method == "xgboost":
            return XGBClassifier(random_state=self.random_state)
        else:
            raise ValueError(
                "Method should be 'logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'svm', 'knn', 'naive_bayes', 'mlp', 'lgbm', 'catboost', or 'xgboost'"
            )

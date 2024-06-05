from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN,
    ClusterCentroids,
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    TomekLinks,
)


class Oversampler:
    """
    Clase que proporciona una interfaz para aplicar técnicas de sobremuestreo.

    Métodos:
    - 'random': Sobremuestreo aleatorio.
    - 'smote': Synthetic Minority Over-sampling Technique.
    - 'adasyn': Adaptive Synthetic Sampling.
    - 'borderlinesmote': Borderline Synthetic Minority Over-sampling Technique.
    - 'svmsmote': Support Vector Machine Synthetic Minority Over-sampling Technique.
    - 'kmeanssmote': KMeans Synthetic Minority Over-sampling Technique.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def provider(self, method, X, y):
        if method == "random":
            resampler = RandomOverSampler(random_state=self.random_state)
        elif method == "smote":
            resampler = SMOTE(random_state=self.random_state)
        elif method == "adasyn":
            resampler = ADASYN(random_state=self.random_state)
        elif method == "borderlinesmote":
            resampler = BorderlineSMOTE(random_state=self.random_state)
        elif method == "svmsmote":
            resampler = SVMSMOTE(random_state=self.random_state)
        elif method == "kmeanssmote":
            resampler = KMeansSMOTE(random_state=self.random_state)
        else:
            raise ValueError(
                "Method should be 'random', 'smote', 'adasyn', 'borderlinesmote', 'svmsmote', or 'kmeanssmote'"
            )

        X_resampled, y_resampled = resampler.fit_resample(X, y)
        return X_resampled, y_resampled


class Undersampler:
    """
    Clase que proporciona una interfaz para aplicar técnicas de submuestreo.


    Métodos:
    - 'random': Submuestreo aleatorio.
    - 'nearmiss': Selección de ejemplos de la clase mayoritaria que están más cerca de la clase minoritaria.
    - 'tomek': Eliminación de enlaces Tomek para limpiar el dataset.
    - 'centroids': Uso de algoritmos de clustering para reducir el tamaño de la clase mayoritaria.
    - 'enn': Edited Nearest Neighbours, elimina ejemplos mal clasificados por sus vecinos más cercanos.
    - 'allknn': Aplica la técnica de eliminación de vecinos más cercanos varias veces.
    """

    def __init__(self, method="random", random_state=42):
        self.method = method
        self.random_state = random_state

    def provider(self, X, y):
        if self.method == "random":
            resampler = RandomUnderSampler(random_state=self.random_state)
        elif self.method == "nearmiss":
            resampler = NearMiss()
        elif self.method == "tomek":
            resampler = TomekLinks()
        elif self.method == "centroids":
            resampler = ClusterCentroids(random_state=self.random_state)
        elif self.method == "enn":
            resampler = EditedNearestNeighbours()
        elif self.method == "allknn":
            resampler = AllKNN()
        else:
            raise ValueError(
                "Method should be 'random', 'nearmiss', 'tomek', 'centroids', 'enn', or 'allknn'"
            )

        X_resampled, y_resampled = resampler.fit_resample(X, y)
        return X_resampled, y_resampled

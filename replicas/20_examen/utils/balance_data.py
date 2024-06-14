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
    - 'RandomOverSampler': Sobremuestreo aleatorio.
    - 'SMOTE': Synthetic Minority Over-sampling Technique.
    - 'ADASYN': Adaptive Synthetic Sampling.
    - 'BorderlineSMOTE': Borderline Synthetic Minority Over-sampling Technique.
    - 'SVMSMOTE': Support Vector Machine Synthetic Minority Over-sampling Technique.
    - 'KMeansSMOTE': KMeans Synthetic Minority Over-sampling Technique.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def provider(self, method, X, y):
        if method == "RandomOverSampler":
            resampler = RandomOverSampler(random_state=self.random_state)
        elif method == "SMOTE":
            resampler = SMOTE(random_state=self.random_state)
        elif method == "ADASYN":
            resampler = ADASYN(random_state=self.random_state)
        elif method == "BorderlineSMOTE":
            resampler = BorderlineSMOTE(random_state=self.random_state)
        elif method == "SVMSMOTE":
            resampler = SVMSMOTE(random_state=self.random_state)
        elif method == "KMeansSMOTE":
            resampler = KMeansSMOTE(random_state=self.random_state)
        else:
            raise ValueError(
                "Method should be 'RandomOverSampler', 'SMOTE', 'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE', or 'KMeansSMOTE'"
            )

        X_resampled, y_resampled = resampler.fit_resample(X, y)
        return X_resampled, y_resampled


class Undersampler:
    """
    Clase que proporciona una interfaz para aplicar técnicas de submuestreo.

    Métodos:
    - 'RandomUnderSampler': Submuestreo aleatorio.
    - 'NearMiss': Selección de ejemplos de la clase mayoritaria que están más cerca de la clase minoritaria.
    - 'TomekLinks': Eliminación de enlaces Tomek para limpiar el dataset.
    - 'ClusterCentroids': Uso de algoritmos de clustering para reducir el tamaño de la clase mayoritaria.
    - 'EditedNearestNeighbours': Edited Nearest Neighbours, elimina ejemplos mal clasificados por sus vecinos más cercanos.
    - 'AllKNN': Aplica la técnica de eliminación de vecinos más cercanos varias veces.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def provider(self, method, X, y):
        if method == "RandomUnderSampler":
            resampler = RandomUnderSampler(random_state=self.random_state)
        elif method == "NearMiss":
            resampler = NearMiss()
        elif method == "TomekLinks":
            resampler = TomekLinks()
        elif method == "ClusterCentroids":
            resampler = ClusterCentroids(random_state=self.random_state)
        elif method == "EditedNearestNeighbours":
            resampler = EditedNearestNeighbours()
        elif method == "AllKNN":
            resampler = AllKNN()
        else:
            raise ValueError(
                "Method should be 'RandomUnderSampler', 'NearMiss', 'TomekLinks', 'ClusterCentroids', 'EditedNearestNeighbours', or 'AllKNN'"
            )

        X_resampled, y_resampled = resampler.fit_resample(X, y)
        return X_resampled, y_resampled

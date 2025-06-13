import os.path
import numpy as np


from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.datasets import load_digits

# TODO: Ajouter les imports nécessaires pour notre algorithme final

#Imports pour le pipeline et la manipulation des données
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

#Imports pour les classes de transformation personnalisées
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.filters import sobel


if os.path.isfile("test_data.npy"):
    X_test = np.load("test_data.npy")
    y_test = np.load("test_labels.npy")
    X_train, y_train = load_digits(return_X_y=True)
else:
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)







# --- DÉBUT CODE ------

# Définition des classes de transformation personnalisées

class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute an average Sobel estimator on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion
    '''
    def __init__(self):
      pass

    def fit(self, X, y=None):
        return self # No fitting needed for this processing

    def transform(self, X):
        sobel_feature = np.array([np.mean(sobel(img.reshape((8,8)))) for img in X]).reshape(-1, 1)
        return sobel_feature



class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute zone information on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion
    '''

    def fit(self, X, y=None):
        return self # No fitting needed for this processing

    def transform(self, X):
        zones=[]
        for img in X:
          img8=img.reshape(8,8)
          zone1_mean=np.mean(img8[:3,:])
          zone2_mean=np.mean(img8[3:5,:])
          zone3_mean=np.mean(img8[5:,:])
          zones.append([zone1_mean,zone2_mean,zone3_mean])
        return np.array(zones)

#Construction du pipeline de caractéristiques
#   On utilise FeatureUnion pour combiner les sorties de plusieurs transformateurs.
#   Les hyperparamètres sont fixés à leurs valeurs optimales trouvées par GridSearchCV.
all_features=FeatureUnion([
    ('pca', PCA(n_components=30)),          # Meilleur n_components
    ('edge', EdgeInfoPreprocessing()),      # Notre extracteur de contours
    ('zones', ZonalInfoPreprocessing())     # Notre extracteur de zones
])

# Ce pipeline final enchaîne l'extraction des caractéristiques, la normalisation, et la classification.
clf=Pipeline([('features', all_features),('scaler', StandardScaler()),('classifier', SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)) # Meilleurs C et gamma trouvés
])



# ------ FIN CODE -------


clf.fit(X_train, y_train)
print(f"Score on the test set {clf.score(X_test, y_test)}")

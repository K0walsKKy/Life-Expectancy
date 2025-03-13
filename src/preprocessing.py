import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from scipy import stats

def load_data(file_path):
    """Charge les données depuis le CSV et remplace quelques valeurs manquantes."""
    df = pd.read_csv(file_path)
    df['Life expectancy'].fillna(df['Life expectancy'].median(), inplace=True)
    df['Adult Mortality'].fillna(df['Adult Mortality'].median(), inplace=True)
    return df

def prepare_data(df):
    """Sépare les caractéristiques (X) et la cible (y)."""
    X = df.drop(columns=['Country', 'Status', 'Life expectancy'])
    y = df['Life expectancy']
    return X, y

def split_and_impute(X, y, test_size=0.2, random_state=13):
    """Sépare en train/test et applique une imputation KNN."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    imputer = KNNImputer(n_neighbors=3)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    return X_train_imp, X_test_imp, y_train, y_test

def transform_features(X_train, X_test):
    """Applique une transformation logarithmique et boxcox sur certaines colonnes."""
    # Transformation logarithmique
    cols_to_transform = ['infant deaths', 'percentage expenditure', 'Measles', 
                         'under-five deaths', 'GDP', 'Population', 
                         'thinness  1-19 years', 'thinness 5-9 years']
    X_train[cols_to_transform] = X_train[cols_to_transform].apply(np.log1p)
    X_test[cols_to_transform]  = X_test[cols_to_transform].apply(np.log1p)
    
    # Transformation Boxcox
    cols_to_box = ['Hepatitis B', 'Polio', 'Diphtheria', 'HIV/AIDS']
    X_train[cols_to_box] = X_train[cols_to_box].apply(lambda x: stats.boxcox(x + 1)[0])
    X_test[cols_to_box]  = X_test[cols_to_box].apply(lambda x: stats.boxcox(x + 1)[0])
    
    return X_train, X_test

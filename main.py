import os
from preprocessing import load_data, prepare_data, split_and_impute, transform_features
from model import train_model, evaluate_model

def main():
    # Chemin vers les données
    file_path = os.path.join("data", "Life Expectancy Data.csv")
    
    # Chargement et préparation des données
    df = load_data(file_path)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_and_impute(X, y)
    X_train, X_test = transform_features(X_train, X_test)
    
    # Entraînement du modèle
    model, random_search = train_model(X_train, y_train)
    
    # Évaluation du modèle
    evaluate_model(model, random_search, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()

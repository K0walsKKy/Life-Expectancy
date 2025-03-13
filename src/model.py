from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform

def train_model(X_train, y_train):
    """Recherche des meilleurs hyperparamètres et entraîne un modèle XGBoost."""
    xgb = XGBRegressor()
    param_dist = {
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(4, 6),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 2),
        'reg_alpha': uniform(0.1, 1),
        'reg_lambda': uniform(0.1, 2)
    }
    random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist,
                                       n_iter=10, cv=3, n_jobs=-1, verbose=1,
                                       scoring='neg_mean_squared_error', random_state=13)
    random_search.fit(X_train, y_train)
    print("Meilleurs paramètres :", random_search.best_params_)
    
    # Entraîne le modèle final avec les meilleurs paramètres
    best_model = XGBRegressor(**random_search.best_params_)
    best_model.fit(X_train, y_train)
    return best_model, random_search

def evaluate_model(model, random_search, X_train, y_train, X_test, y_test):
    """Calcule et affiche les métriques R² et MSE."""
    y_train_pred = random_search.predict(X_train)
    y_test_pred = random_search.predict(X_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    print("R² train :", r2_train)
    print("R² test  :", r2_test)
    print("MSE train :", mse_train)
    print("MSE test  :", mse_test)

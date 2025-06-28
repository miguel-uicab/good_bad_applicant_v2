from sklearn.metrics import precision_recall_curve, auc
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from category_encoders.count import CountEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
import mlflow
import pickle
import logging
import argparse
import platform
from mlflow.tracking import MlflowClient
import yaml
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed data for training")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()

# -----------------------------
# Load model from config
# -----------------------------
def get_model_instance(name, params):
    model_map = {
                'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
                'RandomForestClassifier': RandomForestClassifier,
                'GradientBoostingClassifier': GradientBoostingClassifier,
                'LogisticRegression': LogisticRegression,
                'DecisionTreeClassifier':  DecisionTreeClassifier,
                'XGBClassifier': XGBClassifier
               }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)

# -----------------------------
# Main logic
# -----------------------------
def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg['name'])

    # Configurations
    df = pickle.load(open(args.data, 'rb'))
    objective_name = model_cfg['target_variable']
    list_numeric_names = model_cfg['list_numeric_features']
    list_categorical_names = model_cfg['list_categorical_features']
    seed = 5000
    ratio_balance = 1
    k_folds = 3
    verbose = 10
    test_size = 0.25
    scores = {'f1': 'f1',
             'precision': 'precision',
             'recall': 'recall',
             'm_c': make_scorer(matthews_corrcoef)}
    c_v = StratifiedKFold(n_splits=k_folds,
                            shuffle=True,
                            random_state=seed)
   

    # Use all features except the target variable
    features = df[list_numeric_names + list_categorical_names]
    label = df[objective_name]
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                       label,
                                                       random_state=seed,
                                                       test_size=test_size,
                                                       stratify=label)

    # Get model
    model = get_model_instance(model_cfg['best_model'], model_cfg['parameters'])

    # Start MLflow run
    with mlflow.start_run(run_name="final_training"):
        logger.info(f"Training model: {model_cfg['best_model']}")
        # CONFIGURACIONES GENERALES. ###########
        numeric_transformer = Pipeline(steps=[('scaler',
                                                StandardScaler())])
        categorical_transformer = Pipeline(steps=[('CountEncoder',
                                                    CountEncoder(normalize=True))])
        preprocessor = ColumnTransformer(remainder='passthrough',
                                            transformers=[('numeric',
                                                            numeric_transformer,
                                                            list_numeric_names),
                                                        ('categorical',
                                                            categorical_transformer,
                                                            list_categorical_names)])
        train_transform = Pipeline(steps=[('processing',
                                        preprocessor),
                                    ('RandomUnderSampler',
                                        RandomUnderSampler(random_state=seed,
                                                            sampling_strategy=ratio_balance)),
                                    ('estimator',
                                        model)])
        
        train_transform.fit(X_train, y_train)
        y_pred = train_transform.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        f_1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        matthews_corr = matthews_corrcoef(y_test, y_pred)
        a_s = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        cv_scores = cross_validate(train_transform, 
                                    X_train, 
                                    y_train, 
                                    cv=c_v,
                                    scoring=scores,
                                    return_train_score=False)
        

        # Log params and metrics
        mlflow.log_params(model_cfg['parameters'])
        mlflow.log_metrics({'f1_test': f_1,
                            'recall_test': precision,
                            'precision_test': recall,
                            'm_c_test': matthews_corr,
                            'accuracy_test': a_s,
                            'roc_auc_test': roc_auc,
                            'f1_cv': cv_scores['test_f1'].mean(),
                            'recall_cv': cv_scores['test_recall'].mean(),
                            'precision_cv': cv_scores['test_precision'].mean()})

        # Log and register model
        mlflow.sklearn.log_model(model, "tuned_model")
        # Guarda el modelo entrenado dentro del run, como artefacto.
        #"tuned_model" será la carpeta dentro del experimento que contiene archivos como:
        model_name = model_cfg['name']
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tuned_model"
        # model_name: el nombre del modelo que definiste en tu YAML (ej. "house_price_model").
        # model_uri: la ruta interna del modelo guardado dentro del run.

        logger.info("Registering model to MLflow Model Registry...")
        client = MlflowClient()
        # Usa la clase MlflowClient para interactuar con el Model Registry.
        # Intenta crear el modelo (si no existe aún).
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass  # already exists

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )
        # Crea una nueva versión del modelo (ej. versión 1, 2, etc.), apuntando al modelo guardado en este run.
        # Transition model to "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        #Mueve el modelo a la etapa "Staging", lo que significa: listo para pruebas.
        #Podrías usar "Production" más adelante si quieres marcarlo como oficial.
        # Add a human-readable description
        description = (
            f"Model for predicting house prices.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: All features in the dataset except the target variable\n"
            f"Target variable: {objective_name}\n"
            f"Trained on dataset: {args.data}\n"
            f"Model saved at: {args.models_dir}/trained/{model_name}.pkl\n"
            f"Performance metrics:\n"
            f"  - F1: {f_1:.2f}\n"
            f"  - PRECISION: {precision:.4f}\n"
            f"  - RECALL: {recall:.4f}"
        )
        client.update_registered_model(name=model_name, description=description)

        # Add tags for better organization
        client.set_registered_model_tag(model_name, "algorithm", model_cfg['best_model'])
        client.set_registered_model_tag(model_name, "hyperparameters", str(model_cfg['parameters']))
        client.set_registered_model_tag(model_name, "features", "All features except target variable")
        client.set_registered_model_tag(model_name, "target_variable", label)
        client.set_registered_model_tag(model_name, "training_dataset", args.data)
        client.set_registered_model_tag(model_name, "model_path", f"{args.models_dir}/trained/{model_name}.pkl")

        # Add dependency tags
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in deps.items():
            client.set_registered_model_tag(model_name, k, v)

        # Save model locally
        save_path = f"{args.models_dir}/trained/{model_name}.pkl"
        # joblib.dump(model, save_path)
        pickle.dump(train_transform, open(save_path, 'wb'))
        logger.info(f"Saved trained model to: {save_path}")
        logger.info(f"Final F1: {f_1:.2f}, PRECISION: {precision:.4f}, RECALL: {recall:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

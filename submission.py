import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import optuna
from loguru import logger

FOLDS = 5
NUM_TRIALS = 25
SUBMISSION_ID = time.strftime("%Y%m%d-%H%M%S")


def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df[feature1] * df[feature2]
    return df_new


def objective(trial, x_train, y_train):
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "device": "cuda",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    logger.info(f"Trial {trial.number} parameters: {param}")

    inner_cv = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(x_train.shape[0])

    for i, (train_idx, val_idx) in enumerate(inner_cv.split(x_train, y_train)):
        # print(f"Trial {trial.number} Inner CV fold {i}")
        x_train_inner = x_train.iloc[train_idx]
        y_train_inner = y_train.iloc[train_idx]
        x_val_inner = x_train.iloc[val_idx]
        y_val_inner = y_train.iloc[val_idx]

        model = xgb.train(
            param,
            xgb.DMatrix(x_train_inner, label=y_train_inner),
            num_boost_round=100000,
            evals=[(xgb.DMatrix(x_val_inner, label=y_val_inner), "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        preds = model.predict(xgb.DMatrix(x_val_inner))
        oof[val_idx] = preds

    rmse = np.sqrt(mean_squared_error(y_train, oof))
    logger.info(f"RMSE for trial {trial.number}: {rmse:.4f}")
    return rmse


def submission_pipeline(train_file_path, test_file_path, create_submission=False):
    # Load the dataset
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    # Identifying numerical and categorical columns
    numerical_cols = (
        train_data.drop(columns=["Calories"])
        .select_dtypes(include=["number"])
        .columns.tolist()
    )
    categorical_cols = (
        train_data.drop(columns=["Calories"])
        .select_dtypes(include=["object"])
        .columns.tolist()
    )

    logger.info("Numerical columns: %s", numerical_cols)
    logger.info("Categorical columns: %s", categorical_cols)

    # train_data = add_feature_cross_terms(train_data, numerical_cols)
    # test_data = add_feature_cross_terms(test_data, numerical_cols)

    X = train_data.drop(columns=["id", "Calories"])
    y = np.log1p(train_data["Calories"])
    X_test = test_data.drop(columns=["id"])

    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Building the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(sparse_output=False), categorical_cols),
        ]
    ).set_output(transform="pandas")

    preprocessor.fit(X)
    X = preprocessor.transform(X)
    X_test = preprocessor.transform(X_test)

    outer_cv = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    oof = np.zeros(X.shape[0])
    pred = np.zeros(X_test.shape[0])

    for i, (train_idx_outer, valid_idx_outer) in enumerate(outer_cv.split(X, y)):
        logger.info(f"\n {'#' * 10} Fold {i+1} {'#'*10}")
        start = time.time()

        x_train_outer = X.iloc[train_idx_outer].copy()
        y_train_outer = y.iloc[train_idx_outer]
        x_val_outer = X.iloc[valid_idx_outer].copy()
        y_val_outer = y.iloc[valid_idx_outer]
        x_test = X_test.copy()

        # Optimizing hyperparameters
        study = optuna.create_study(
            storage="sqlite:///data/optuna_study.db",
            direction="minimize",
            study_name=f"{SUBMISSION_ID}_{i+1}",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective(trial, x_train_outer, y_train_outer),
            n_trials=NUM_TRIALS,
            n_jobs=10,
        )
        best_hparams_for_fold = study.best_params
        logger.info("Best hyperparameters for fold: %s", best_hparams_for_fold)
        logger.info("Best RMSE for fold: %s", study.best_value)

        # Training the model
        model = xgb.train(
            best_hparams_for_fold,
            xgb.DMatrix(x_train_outer, label=y_train_outer),
            num_boost_round=100000,
            evals=[(xgb.DMatrix(x_val_outer, label=y_val_outer), "val")],
            early_stopping_rounds=100,
            verbose_eval=100,
        )
        oof[valid_idx_outer] = model.predict(xgb.DMatrix(x_val_outer))

        # Making predictions
        preds = model.predict(xgb.DMatrix(x_test))
        pred += preds

        rmse = np.sqrt(mean_squared_error(y_val_outer, oof[valid_idx_outer]))
        logger.info(f"Fold {i+1} RMSE: {rmse:.4f}")
        logger.info(f"Time: {time.time() - start:.2f} seconds")

    pred /= FOLDS

    # Final RMSE
    full_rmse = np.sqrt(mean_squared_error(y, oof))
    logger.info(f"\nFinal CV RMSE: {full_rmse:.4f}")

    # Creating a DataFrame for submission
    if create_submission:
        y_preds = np.expm1(pred)
        y_preds = np.clip(y_preds, 1, 314)
        submission_df = pd.DataFrame({"id": test_data["id"], "Calories": y_preds})

        submission_df.to_csv(
            f"data/submission/submission_{SUBMISSION_ID}.csv", index=False
        )


if __name__ == "__main__":

    train_file_path = "data/raw/train.csv"
    test_file_path = "data/raw/test.csv"

    create_submission = True

    submission_pipeline(train_file_path, test_file_path, create_submission)

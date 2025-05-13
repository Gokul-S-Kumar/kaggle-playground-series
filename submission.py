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

FOLDS = 50


def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df[feature1] * df[feature2]
    return df_new


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

    print("Numerical columns:", numerical_cols)
    print("Categorical columns:", categorical_cols)

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

    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    oof = np.zeros(X.shape[0])
    pred = np.zeros(X_test.shape[0])

    for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\n {'#' * 10} Fold {i+1} {'#'*10}")

        x_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx]
        x_val = X.iloc[valid_idx].copy()
        y_val = y.iloc[valid_idx]
        x_test = X_test.copy()

        start = time.time()

        model = xgb.XGBRegressor(
            device="cuda",
            max_depth=10,
            colsample_bytree=0.75,
            subsample=0.9,
            n_estimators=2000,
            learning_rate=0.02,
            gamma=0.01,
            max_delta_step=2,
            early_stopping_rounds=100,
            eval_metric="rmse",
            enable_categorical=True,
        )

        # Fitting the model
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)

        # Making predictions
        oof[valid_idx] = model.predict(x_val)
        pred += model.predict(x_test)

        rmse = np.sqrt(mean_squared_error(y_val, oof[valid_idx]))
        print(f"Fold {i+1} RMSE: {rmse:.4f}")
        print(f"Time: {time.time() - start:.2f} seconds")

    pred /= FOLDS

    # Final RMSE
    full_rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"\nFinal CV RMSE: {full_rmse:.4f}")

    # Creating a DataFrame for submission
    if create_submission:
        y_preds = np.expm1(pred)
        y_preds = np.clip(y_preds, 1, 314)
        submission_df = pd.DataFrame({"id": test_data["id"], "Calories": y_preds})
        submission_id = time.strftime("%Y%m%d-%H%M%S")
        submission_df.to_csv(
            f"data/submission/submission_{submission_id}.csv", index=False
        )


if __name__ == "__main__":

    train_file_path = "data/raw/train.csv"
    test_file_path = "data/raw/test.csv"

    create_submission = True

    submission_pipeline(train_file_path, test_file_path, create_submission)

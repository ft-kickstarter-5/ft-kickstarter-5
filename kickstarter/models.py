from category_encoders import OrdinalEncoder
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def wrangle(df):
    df = df.drop(
        columns=[
            "id",
            "usd_pledged",
            "spotlight",
            "photo",
            "state",
            "name",
            "blurb",
            "slug",
            "Unnamed: 0",
            "creator",
            "location",
            "profile",
            "urls",
            "source_url",
            "friends",
            "is_starred",
            "is_backing",
            "permissions",
            "name_len",
            "name_len_clean",
        ]
    )
    return df


def create_model(filename):
    df = pd.read_csv(filename)
    df = wrangle(df)

    col = df.columns
    test_col = df[
        ["goal", "category", "staff_pick", "state_changed_at_month", "SuccessfulBool"]
    ]

    pd.DataFrame(test_col)

    target = "SuccessfulBool"
    # y= df[target]
    # X = df.drop(columns=target)
    y = test_col[target]
    X = test_col.drop(columns=target)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # Decision Tree
    model_dt = make_pipeline(
        OrdinalEncoder(),
        SimpleImputer(strategy="mean"),
        DecisionTreeClassifier(random_state=7),
    )

    model_dt.fit(X_train, y_train)

    # Random Forest
    model_rf = make_pipeline(
        OrdinalEncoder(),
        SimpleImputer(),
        RandomForestClassifier(
            random_state=7, n_estimators=15, max_depth=8, min_samples_leaf=2
        ),
    )

    model_rf.fit(X_train, y_train)

    y_train.value_counts(normalize=True).max()

    print("Training Accuracy:", model_rf.score(X_train, y_train))
    print("Validation Accuracy:", model_rf.score(X_val, y_val))

    model_rf.score(X_val, y_val)

    return model_rf


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "data/kickstarter_data_full.csv")
    final_model = create_model(filename)
    pickle.dump(final_model, open(os.path.join(dirname, "final-model"), "wb"))
else:
    final_model = create_model("./data/kickstarter_data_full.csv")
    pickle.dump(final_model, open(os.path.join("./final-model"), "wb"))

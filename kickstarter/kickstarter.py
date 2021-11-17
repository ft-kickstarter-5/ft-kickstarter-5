import pandas as pd
import numpy as np
from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pickle


def wrangle(data_path):
    """wrangle the data, clean data"""

    # turn all column headers to use same format
    df = pd.read_csv(data_path)
    return df


df = wrangle("./kickstarter/data/kickstarter_data_full.csv")

# target is SuccessfulBool
df["SuccessfulBool"].value_counts()

"""Split dataset into train/test"""
target = "SuccessfulBool"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
y_train, y_test = train_test_split(y, test_size=0.1, random_state=42)

"""Rush to baseline"""
baseline_predictions = [y_train.value_counts(ascending=False).index[0]] * len(X_train)
baseline_accuracy = accuracy_score(y_train, baseline_predictions)


"""Logistic Regression"""
logr_pipeline = make_pipeline(
    OrdinalEncoder(), SimpleImputer(), LogisticRegression(random_state=42)
)
logr_model = logr_pipeline.fit(X_train, y_train)

logr_train_predictions = logr_model.predict(X_train)
logr_train_accuracy = accuracy_score(y_train, logr_train_predictions)

logr_test_predictions = logr_model.predict(X_test)
logr_test_accuracy = accuracy_score(y_test, logr_test_predictions)

features = logr_model.named_steps.ordinalencoder.get_feature_names()

pd.DataFrame(
    data=logr_model.named_steps.logisticregression.coef_.tolist()[0],
    index=features,
    columns=["coefficient"],
).abs().sort_values(by="coefficient", ascending=False).head(10).plot(kind="barh")
pd.DataFrame(
    data=logr_model.named_steps.logisticregression.coef_.tolist()[0],
    index=features,
    columns=["coefficient"],
).abs().sort_values(by="coefficient", ascending=False)

final_model = logr_model
time = pd.to_datetime("now").strftime("%Y-%m-%d-%H:%M:%S")
filename = time + "-" + final_model.steps[len(final_model.steps) - 1][0]


pickle.dump(final_model, open("./kickstarter/models/" + filename, "wb"))

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


df = pd.read_csv("./data/kickstarter_data_full.csv")
df = wrangle(df)

col = df.columns
test_col = df[['goal','category','staff_pick',
                'state_changed_at_month','SuccessfulBool']]
pd.DataFrame(test_col)

target = "SuccessfulBool"
# y= df[target]
# X = df.drop(columns=target)
y = test_col[target]
X = test_col.drop(columns=target)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)

#Random Forest
model_rf = make_pipeline(
    OrdinalEncoder(), SimpleImputer(),
    RandomForestClassifier(random_state=7,n_estimators=15,
                           max_depth=8,min_samples_leaf=2)
)

model_rf.fit(X_train, y_train)

y_train.value_counts(normalize=True).max()

print("Training Accuracy:", model_rf.score(X_train, y_train))
print(" Validation Accuracy:", model_rf.score(X_val, y_val))

model_rf.score(X_val, y_val)

final_model = model_rf
time = pd.to_datetime("now").strftime("%Y-%m-%d-%H:%M:%S")
filename = time + "-" + final_model.steps[len(final_model.steps) - 1][0]


pickle.dump(final_model, open("./models/" + filename, "wb"))
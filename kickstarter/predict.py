import os
import numpy as np
import pickle
import pandas as pd

final_model = pickle.load(open("./kickstarter/final-model", "rb"))


def predict_success(feature_inputs):
    return final_model.predict(
        [
            pd.Series(
                [
                    feature_inputs["goal"],
                    feature_inputs["category"],
                    feature_inputs["staff_pick"],
                    bool(int(feature_inputs["state_changed_at_month"])),
                ],
                index=[
                    "goal",
                    "category",
                    "staff_pick",
                    "state_changed_at_month",
                ],
            )
        ]
    )


def model_importances():
    return final_model.coef_

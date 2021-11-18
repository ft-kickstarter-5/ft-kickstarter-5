import os
import numpy as np
import pickle
import pandas as pd

final_model = pickle.load(open("./kickstarter/final-model", "rb"))


def predict_success(feature_inputs):
    print(feature_inputs["category"])
    return final_model.predict(
        [
            pd.Series(
                [
                    feature_inputs["goal"],
                    feature_inputs["pledged"],
                    feature_inputs["launch_to_deadline_days"],
                    feature_inputs["launch_to_state_change_days"],
                    feature_inputs["backers_count"],
                    feature_inputs["category"],
                ],
                index=[
                    "goal",
                    "pledged",
                    "launch_to_deadline_days",
                    "launch_to_state_change_days",
                    "backers_count",
                    "category",
                ],
            )
        ]
    )


def model_importances():
    return final_model.coef_

import os
import numpy as np
import pickle
import pandas as pd

final_model = pickle.load(
    open("./kickstarter/models/2021-11-17-02:15:14-randomforestclassifier", "rb")
)


def predict_success(feature_inputs):
    return final_model.predict(
        [
            pd.Series(
                [1, 4, 3, 6, 7, "Web"],
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

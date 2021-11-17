import os
import numpy as np
import pickle


final_model = pickle.load(
    open("./kickstarter/models/2021-11-16-22:56:47-logisticregression", "rb")
)


def predict_success(feature_inputs):
    return final_model.predict([feature1, feature2])


def model_importances():
    return final_model.coef_

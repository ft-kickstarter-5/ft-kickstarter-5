from flask import Flask, render_template, request
from .kickstarter import df
from .predict import predict_success


def create_app():

    # initializes our app
    app = Flask(__name__)

    # Listen to a "route"
    # '/' is the home page route
    @app.route("/")
    @app.route("/index")
    def root():
        # query the db for all users
        campaigns = df
        # what I want to happen when somebody goes to the home page
        return render_template("index.html", title="Home", campaigns=campaigns)

    @app.route("/landing")
    def landing():
        # query the db for all users
        # what I want to happen when somebody goes to the home page
        return render_template("landing.html", title="Landing")

    @app.route("/generic")
    def generic():
        # query the db for all users
        # what I want to happen when somebody goes to the home page
        return render_template("generic.html", title="Generic")

    @app.route("/predict", methods=["POST", "GET"])
    def predict():
        # query the db for all users
        # what I want to happen when somebody goes to the home page
        if request.method == "POST":
            model_inputs = request.form
            print(model_inputs)
            prediction = predict_success(model_inputs)
            return render_template(
                "result.html",
                title="Prediction",
                prediction=prediction,
                inputs=model_inputs,
            )
        else:
            return render_template("form.html", title="Form")

    return app

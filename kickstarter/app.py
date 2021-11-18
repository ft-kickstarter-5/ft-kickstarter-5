from flask import Flask, render_template, request
from .kickstarter import df
from .predict import predict_success


def create_app():
    # initializes our flask app
    app = Flask(__name__)

    # Listen to a "route"
    # '/' is the home page route
    @app.route("/")
    @app.route("/index")
    def root():
        """returns template for home page of flask app"""
        return render_template("index.html", title="Home")

    @app.route("/landing")
    def landing():
        """placeholder for landing page in forty template"""
        return render_template("landing.html", title="Landing")

    @app.route("/elements")
    def elements():
        """placeholder for elements page in forty template"""
        return render_template("elements.html", title="Elements")

    @app.route("/generic")
    def generic():
        """placeholder for generic page in forty template"""
        return render_template("generic.html", title="Generic")

    @app.route("/predict", methods=["POST", "GET"])
    def predict():
        """Returns a form page on initial get request.
        After post, returns results html with predictions from model"""
        if request.method == "POST":
            model_inputs = request.form
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

if __name__ == '__main__':
    app.run()
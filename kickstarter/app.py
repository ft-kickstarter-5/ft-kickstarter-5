from flask import Flask, render_template, request
from .kickstarter import df


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

    @app.route("/elements")
    def elements():
        # query the db for all users
        # what I want to happen when somebody goes to the home page
        return render_template("elements.html", title="Home")

    return app

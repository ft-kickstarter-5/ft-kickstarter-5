from flask import Flask, render_template, request
from .kickstarter import df


def create_app():

    # initializes our app
    app = Flask(__name__)

    # Listen to a "route"
    # '/' is the home page route
    @app.route("/")
    def root():
        # query the db for all users
        campaigns = df
        # what I want to happen when somebody goes to the home page
        return render_template("index.html", title="Home", campaigns=campaigns)

    return app

from flask import current_app
from flask import Blueprint, jsonify, request
from ..models.complevel_predictor import complevel_predictor
from ..models.complevel_model_trainer import complevel_model_trainer
import requests

complevel_blueprint = Blueprint('complevel', __name__)

@complevel_blueprint.route("/predictCompLevel", methods=["POST"])
def predict_complevel():
    data = request.get_json()
    title = data["title"]
    description = data["description"]
    complevelmodel = complevel_predictor()
    prediction = complevelmodel.predict(title, description)

    return jsonify(prediction)


@complevel_blueprint.route("/trainCompLevel", methods=["POST"])
def train_complevel():
    data = request.get_json()
    trainer = complevel_model_trainer()
    training_stats = trainer.train(data)
    return jsonify(training_stats)


@complevel_blueprint.route("/getCompLevelReport", methods=["GET"])
def report_complevel():
    trainer = complevel_model_trainer()
    report = trainer.getReport()
    return jsonify(report)
from flask import Blueprint, jsonify, request

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route("/", methods=["GET"])
def index():
    return jsonify({"about": "This is an API providing AI-predictions for identifying learning outcomes in course descriptions."})

@main_blueprint.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})
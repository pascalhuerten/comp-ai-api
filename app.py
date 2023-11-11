from flask import Flask
from src.views.main import main_blueprint
from src.views.skills import skills_blueprint
from src.setup import setup

app = Flask(__name__)
app.register_blueprint(main_blueprint)
app.register_blueprint(skills_blueprint)

# Initialize resources
with app.app_context():
    embedding, skilldbs, skillfit_model, storage = setup()

# Store resources in app's config so they can be accessed in views
app.config['EMBEDDING'] = embedding
app.config['SKILLDBS'] = skilldbs
app.config['SKILLFIT_MODEL'] = skillfit_model
app.config['storage'] = storage

if __name__ == "__main__":
    app.run(debug=True)

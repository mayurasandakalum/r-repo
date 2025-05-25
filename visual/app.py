from flask import Flask, render_template, jsonify
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from kinesthetic.models import User
import sys
import os

# Fix the import mechanism for the config module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Use insert instead of append to prioritize this path

try:
    import config
except ImportError:
    print(f"Error: Cannot import config module. Looking in: {parent_dir}")
    print(f"Python path: {sys.path}")
    raise

login_manager = LoginManager()
csrf = CSRFProtect()

app = Flask(__name__)
# app.config.from_object(Config)
app.config['SECRET_KEY'] = config.SECRET_KEY

# Initialize CSRF protection
csrf.init_app(app)

# Initialize login manager with correct login view
login_manager.init_app(app)
login_manager.login_view = (
    "kinesthetic.login"  # Changed from "login" to "kinesthetic.login"
)

from kinesthetic.routes import kinesthetic_blueprint
from kinesthetic.utils import load_initial_questions

app.register_blueprint(kinesthetic_blueprint)

# Load initial questions if they don't exist
with app.app_context():
    load_initial_questions()

@app.errorhandler(404)
def not_found(e):
    return render_template("error_404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("error_500.html"), 500

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)

@app.route('/api/info')
def api_info():
    return jsonify({
        'app': 'Visual App',
        'status': 'running'
    })

if __name__ == "__main__":
    app.run(debug=True, port=config.VISUAL_APP_PORT)

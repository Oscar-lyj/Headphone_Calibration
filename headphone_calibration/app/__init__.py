from flask import Flask
import os
from app.routes import init_routes

def create_app():
    app = Flask(__name__)
    #app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
    app.config['UPLOAD_FOLDER'] = os.path.join('app', 'static', 'uploads')
    app.secret_key = 'your_secret_key'
    
    # Initialize routes
    init_routes(app)

    return app
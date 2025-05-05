from flask import Flask
from app.config import Config
from app.extensions import cors, db, migrate
from app.errors import register_error_handlers
from app.routes.vectordb import vectordb_bp
from app.routes.chat import chat_bp
from app.routes.returnpdf import returnpdf_bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app)

    # Register blueprints & error handlers
    app.register_blueprint(vectordb_bp, url_prefix='/api/vectordb')
    app.register_blueprint(chat_bp,url_prefix='/api/chat')
    app.register_blueprint(returnpdf_bp,url_prefix='/api/returnpdf')
    register_error_handlers(app)
    
    return app

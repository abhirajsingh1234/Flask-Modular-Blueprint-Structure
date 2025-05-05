import os
import urllib   
from dotenv import load_dotenv

load_dotenv()
db_driver = os.getenv("DB_DRIVER")
db_server = os.getenv("DB_SERVER")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

db_params = urllib.parse.quote_plus(
    f'DRIVER={{{db_driver}}};'
    f'SERVER={db_server};'
    f'DATABASE={db_name};'
    f'UID={db_user};'
    f'PWD={db_password}'
)

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-me')
    SQLALCHEMY_DATABASE_URI = f"mssql+pyodbc:///?odbc_connect={db_params}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_HEADERS = 'Content-Type'

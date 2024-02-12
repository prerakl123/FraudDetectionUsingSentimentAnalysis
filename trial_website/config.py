import os


class Config:
    FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")
    # SERVER_NAME = 'videosentimentanalysis.com:5500'
    SERVER_NAME = 'localhost:5500'

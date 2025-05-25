import os
import secrets

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    # Update the path to match models.py
    FIREBASE_CREDENTIALS = os.path.join(basedir, "serviceAccountKey.json")
    DEBUG = True

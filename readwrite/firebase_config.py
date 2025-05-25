import firebase_admin
from firebase_admin import credentials, firestore, auth
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to your service account key JSON file - using absolute path
service_account_path = os.path.join(
    current_dir, "learn-pal-firebase-adminsdk-ugedp-fcb865a7d8.json"
)

# Initialize Firebase
try:
    # Try to get the existing app
    firebase_app = firebase_admin.get_app("readwrite")
except ValueError:
    # If no app exists with this name, initialize a new one
    # First check if the file exists
    if not os.path.exists(service_account_path):
        print(f"ERROR: Service account file not found at: {service_account_path}")
        # Use a fallback approach - try to find it in parent directory
        parent_dir = os.path.dirname(current_dir)
        service_account_path = os.path.join(
            parent_dir, "readwrite", "learn-pal-firebase-adminsdk-ugedp-fcb865a7d8.json"
        )
        if not os.path.exists(service_account_path):
            print(
                f"ERROR: Service account file not found in alternative location either: {service_account_path}"
            )
            raise FileNotFoundError(f"Could not find Firebase credentials file.")
        else:
            print(
                f"Found service account in alternative location: {service_account_path}"
            )

    print(f"Initializing Firebase app with credentials from: {service_account_path}")
    cred = credentials.Certificate(service_account_path)
    firebase_app = firebase_admin.initialize_app(cred, name="readwrite")

# Access Firestore
db = firestore.client(app=firebase_app)


def verify_token(token):
    """Verify a Firebase token and return the decoded token."""
    try:
        decoded_token = auth.verify_id_token(token, app=firebase_app)
        return decoded_token
    except Exception as e:
        print(f"Error verifying token: {e}")
        return None

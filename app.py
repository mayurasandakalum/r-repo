"""
Main app entry point that launches all three Flask applications.
"""

import subprocess
import sys
import os
import time
import threading
import webbrowser
from flask import Flask
from flask_wtf.csrf import CSRFProtect
import config
from routes import init_routes

# Initialize the main Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY

# Initialize CSRF protection BEFORE routes
# csrf = CSRFProtect()
# csrf.init_app(app)

# Now initialize routes
print("Initializing routes...")
init_routes(app)
print("Routes initialized!")

# Exempt specific routes if needed
# csrf.exempt("kinesthetic.mark_video_as_watched")

# Store the process objects for cleanup
app_processes = []


def launch_app(app_path, port):
    """Launch a Flask app as a separate process."""
    process = subprocess.Popen([sys.executable, app_path])
    app_processes.append(process)
    print(f"Launched app {app_path} on port {port}")
    return process


def wait_for_app(url, max_retries=10):
    """Wait for an app to be ready by checking its URL."""
    for i in range(max_retries):
        try:
            import requests

            response = requests.get(url)
            if response.status_code == 200:
                print(f"App at {url} is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print(f"App at {url} did not start properly")
    return False


def create_templates_folder():
    """Create templates folder if it doesn't exist."""
    os.makedirs("templates", exist_ok=True)


def cleanup():
    """Clean up all processes on exit."""
    for process in app_processes:
        try:
            process.terminate()
            print(f"Terminated process {process.pid}")
        except:
            pass


def open_browser():
    """Open browser to the main app after a short delay."""
    time.sleep(2)
    webbrowser.open(f"http://localhost:{config.MAIN_APP_PORT}")


if __name__ == "__main__":
    # Create necessary templates
    create_templates_folder()

    # Register cleanup function
    import atexit

    atexit.register(cleanup)

    # Launch the other apps
    kinesthetic_process = launch_app("kinesthetic/app.py", config.KINESTHETIC_APP_PORT)
    readwrite_process = launch_app("readwrite/app.py", config.READWRITE_APP_PORT)
    visual_process = launch_app("visual/app.py", config.VISUAL_APP_PORT)
    audio_process = launch_app("audio/app.py", config.AUDIO_APP_PORT)

    # Wait for the apps to be ready
    kinesthetic_url = f"http://localhost:{config.KINESTHETIC_APP_PORT}"
    readwrite_url = f"http://localhost:{config.READWRITE_APP_PORT}"
    visual_url = f"http://localhost:{config.VISUAL_APP_PORT}"
    audio_url = f"http://localhost:{config.AUDIO_APP_PORT}"

    # Start a thread to open the browser when the apps are ready
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Run the main app
    app.run(debug=True, port=config.MAIN_APP_PORT, use_reloader=False)

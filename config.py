"""
Shared configuration for all Flask apps.
"""

# Secret key for session signing (should be a random string in production)
SECRET_KEY = 'your-secret-key-here'

# Ports for each app
MAIN_APP_PORT = 5000
KINESTHETIC_APP_PORT = 5001
READWRITE_APP_PORT = 5002
VISUAL_APP_PORT = 5003  # New port for visual component
AUDIO_APP_PORT = 5004
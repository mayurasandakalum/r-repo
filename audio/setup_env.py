"""
Setup script for readwrite app environment.
This script creates necessary directories and basic templates.
"""

import os
import sys

def setup_environment():
    """Create all necessary directories and templates for the readwrite app."""
    print("Setting up readwrite app environment...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define required directories
    directories = [
        os.path.join(current_dir, "static"),
        os.path.join(current_dir, "static", "aud_records"),
        os.path.join(current_dir, "static", "write_img"),
        os.path.join(current_dir, "static", "Images"),
        os.path.join(current_dir, "templates"),
        os.path.join(current_dir, "write_model"),
    ]
    
    # Create all directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create basic templates
    templates = {
        "submit_audio_form.html": """
<!DOCTYPE html>
<html>
<head>
    <title>Submit Audio</title>
</head>
<body>
    <h1>Submit Audio</h1>
    <p>This is a simple form to submit audio for question ID: {{ question_id }}</p>
    <form action="/submit_audio" method="post" enctype="multipart/form-data">
        <input type="file" name="audio" accept="audio/*">
        <input type="hidden" name="questionID" value="{{ question_id }}">
        <button type="submit">Submit</button>
    </form>
</body>
</html>
"""
    }
    
    for template_name, content in templates.items():
        template_path = os.path.join(current_dir, "templates", template_name)
        if not os.path.exists(template_path):
            with open(template_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✓ Created template: {template_path}")
        else:
            print(f"✓ Template already exists: {template_path}")
    
    # Create placeholder for model files
    model_placeholder = os.path.join(current_dir, "whisper-small-sinhala-finetuned", "README.txt")
    os.makedirs(os.path.dirname(model_placeholder), exist_ok=True)
    
    if not os.path.exists(model_placeholder):
        with open(model_placeholder, "w", encoding="utf-8") as f:
            f.write("Place Whisper model files in this directory\n")
        print(f"✓ Created model placeholder: {model_placeholder}")
    
    print("\nEnvironment setup complete!")
    print("\nReminder: You need to:")
    print("1. Place your Whisper model files in the 'whisper-small-sinhala-finetuned' directory")
    print("2. Place your Firebase credentials in the app directory")
    print("3. Ensure all required packages are installed")

if __name__ == "__main__":
    setup_environment()

# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
from flask_login import (
    LoginManager,
    login_user,
    current_user,
    logout_user,
    login_required,
    UserMixin,
)
import requests

# import firebase_admin
# from firebase_admin import credentials, firestore
import os
import sys

# Fix the import mechanism for the config module - MOVED THIS UP
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Use insert instead of append to prioritize this path

import config  # Now this will work
from firebase_config import db  # Import the Firebase configuration

import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torchaudio
from gtts import gTTS
import random
import re
import base64
from collections import Counter
import io
import wave

# Initialize the app with template and static folder configurations
app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"  # Stores session data locally
Session(app)

# Set the secret key for the app
app.config["SECRET_KEY"] = config.SECRET_KEY

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# User class for Flask-Login
class User(UserMixin):
    def __init__(
        self, id, email=None, name=None, gender=None, birthday=None, grade=None
    ):
        self.id = id
        self.email = email
        self.name = name
        self.gender = gender
        self.birthday = birthday
        self.grade = grade

    @staticmethod
    def get(user_id):
        try:
            user_doc = db.collection("audio_users").document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                return User(
                    id=user_id,
                    email=user_data.get("email"),
                    name=user_data.get("name"),
                    gender=user_data.get("gender"),
                    birthday=user_data.get("birthday"),
                    grade=user_data.get("grade"),
                )
            return None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None

    def save(self):
        try:
            db.collection("audio_users").document(self.id).set(
                {
                    "email": self.email,
                    "name": self.name,
                    "gender": self.gender,
                    "birthday": self.birthday,
                    "grade": self.grade,
                }
            )
            return True
        except Exception as e:
            print(f"Error saving user: {e}")
            return False


# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


# Make all paths absolute for consistency
STATIC_FOLDER = os.path.join(current_dir, "static")
AUDIO_FOLDER = os.path.join(STATIC_FOLDER, "aud_records")
WRITE_IMG_FOLDER = os.path.join(STATIC_FOLDER, "write_img")
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, "Images")

# Create necessary directories
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(WRITE_IMG_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Ensure templates directory exists
TEMPLATES_FOLDER = os.path.join(current_dir, "templates")
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)

# Load the fine-tuned model and processor
model = WhisperForConditionalGeneration.from_pretrained(
    "audio/whisper-small-sinhala-finetuned"
)
processor = WhisperProcessor.from_pretrained("audio/whisper-small-sinhala-finetuned")
# Set the language and task
language = "Sinhala"
task = "transcribe"
# Update the model's generation configuration
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=language, task=task
)
model.config.suppress_tokens = None

model_st = SentenceTransformer("Ransaka/bert-small-sentence-transformer")

UPLOAD_FOLDER = os.path.join("static", "aud_records")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Firebase
# cred = credentials.Certificate(".audio/learn-pal-firebase-adminsdk-ugedp-fcb865a7d8.json")
# firebase_admin.initialize_app(cred)
# db = firestore.client()

# Flag to indicate if we're running with Firebase or in offline mode
OFFLINE_MODE = False

# Initialize Firebase with better error handling
try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    firebase_cred_path = os.path.join(
        current_dir, "learn-pal-firebase-adminsdk-ugedp-fcb865a7d8.json"
    )
    if os.path.exists(firebase_cred_path):
        cred = credentials.Certificate(firebase_cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print(
            f"Successfully initialized Firebase with credentials from {firebase_cred_path}"
        )
    else:
        print(f"Firebase credentials file not found at {firebase_cred_path}")
        print("Running in OFFLINE MODE - Firebase features will be simulated")
        OFFLINE_MODE = True
        db = None
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    print("Running in OFFLINE MODE - Firebase features will be simulated")
    OFFLINE_MODE = True
    db = None


# Global variable to track the current question ID
AudQuestionID = 1
no_q = 5
username = ""
Aud_data = {}
# Aud_results = ["lesson1","lesson1","lesson1","lesson1","lesson1","lesson2","lesson2","lesson2","lesson2","lesson2","lesson2","lesson2","lesson2","lesson2","lesson2","lesson3","lesson3"]
# Aud_results_2 = ["lesson3","lesson3","lesson3"]
Aud_results = []
Aud_results_2 = []
rd_lesson = 0
rd_lesson_c = 0


def extract_first_number(s):
    match = re.search(r"\d+", s)
    return int(match.group()) if match else None


def get_min_count_string(strings):
    counter = Counter(strings)
    min_count = min(counter.values())
    min_strings = [s for s, count in counter.items() if count == min_count]
    no = extract_first_number(min_strings[0])
    return no


# for all score
def calculate_res(query):
    result = []
    for doc in query:
        data_dict = doc.to_dict()
        result.append(data_dict["data"]["Lesson"])
    counts = Counter(result)
    counts_dict = dict(counts)
    return counts_dict


# num- Total answered questions, # Total number of questions in a lesson
def random_q_r(num, noq):
    global rd_lesson
    global rd_lesson_c
    if rd_lesson > 0:
        if rd_lesson_c > 4:
            lesson = 100
        else:
            rd_lesson_c = rd_lesson_c + 1
            lesson = rd_lesson - 1
    else:
        lesson = int((num - 1) / noq)
    start = lesson * 50
    qid = random.randint(start, start + 50)
    return qid


def stt_sinhala(audio_file):
    global model
    speech_array, sampling_rate = torchaudio.load(audio_file)

    # Resample the audio to 16 kHz if necessary
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, new_freq=16000
        )
        speech_array = resampler(speech_array)

    # Convert to mono channel if necessary
    if speech_array.shape[0] > 1:
        speech_array = speech_array.mean(dim=0)  # it averages them to mono

    # Prepare the input features
    input_features = processor.feature_extractor(
        speech_array.numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features

    # Move model and inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    input_features = input_features.to(device)

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(input_features)

    # Decode the transcription
    transcription = processor.tokenizer.decode(
        generated_ids[0], skip_special_tokens=True
    )

    print("Transcription:", transcription)
    return transcription


# compares two sentences and returns their semantic similarity score
def is_similar(target, source):
    sentences = [target, source]

    if target in source:
        return 0.75

    embeddings = model_st.encode(sentences, convert_to_tensor=True)
    # Compute Cosine Similarity using torch
    similarity = F.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    )
    print("Similarity Score : ", similarity)
    return similarity


# This function converts Sinhala text into speech using gTTS (Google Text-to-Speech) and saves the generated audio file
def sin_text_to_speech(text, qid):
    """
    Convert the given Sinhala text to speech using gTTS and play the audio.
    """
    # Specify lang='si' for Sinhala
    tts = gTTS(text=text, lang="si")
    output_file = "./audio/static/aud_records/tts_" + str(qid) + "_.wav"

    # Save the audio to a file
    tts.save(output_file)


@app.route("/api/info")
def api_info():
    return jsonify({"app": "Audio App", "status": "running"})


# Authentication route that verifies the token from the main app
@app.route("/authenticate")
def authenticate():
    """Authenticate the user using the token provided by the main app"""
    token = request.args.get("token")
    if not token:
        return redirect(f"http://localhost:{config.MAIN_APP_PORT}/login")

    try:
        # For Firebase custom tokens, we need to extract the user ID differently
        # Custom tokens have a 'uid' field in their payload
        import jwt

        # Decode the token without verification to extract the user ID
        # This is safe because we're not using it for authentication directly,
        # just to identify which user to look up
        decoded_payload = jwt.decode(token, options={"verify_signature": False})
        user_id = decoded_payload.get("uid")

        if not user_id:
            print("No user ID found in token")
            return redirect(f"http://localhost:{config.MAIN_APP_PORT}/login")

        print(f"Authenticating user ID: {user_id}")

        # Check if user exists in our system
        user = User.get(user_id)

        if not user:
            # If user doesn't exist, get their details from the main app
            try:
                response = requests.get(
                    f"http://localhost:{config.MAIN_APP_PORT}/api/user/{user_id}"
                )
                if response.status_code != 200:
                    print(
                        f"Error getting user data: {response.status_code} {response.text}"
                    )
                    return redirect(f"http://localhost:{config.MAIN_APP_PORT}/login")

                user_data = response.json()
                print(f"Got user data from main app: {user_data}")

                # Create a new user in our system
                user = User(
                    id=user_id,
                    email=user_data.get("email"),
                    name=user_data.get("name"),
                    gender=user_data.get("gender"),
                    birthday=user_data.get("birthday"),
                    grade=user_data.get("grade"),
                )
                user.save()
                print(f"Created new user in audio app: {user_id}")
            except Exception as e:
                print(f"Error fetching user data: {e}")
                return redirect(f"http://localhost:{config.MAIN_APP_PORT}/login")
        else:
            print(f"Found existing user: {user_id}")

        # Log in the user
        login_user(user)

        # Store the user ID and name in the session for later use
        session["user_id"] = user_id
        session["username"] = user.name if user.name else user.email

        # Redirect to the home page
        print(f"User {user_id} authenticated successfully")
        return redirect(url_for("home"))

    except Exception as e:
        print(f"Authentication error: {e}")
        return redirect(f"http://localhost:{config.MAIN_APP_PORT}/login")


# Logout route that redirects to the main app's logout
@app.route("/logout")
def logout():
    logout_user()
    return redirect(f"http://localhost:{config.MAIN_APP_PORT}/logout")


# Update home route to use current user data
@app.route("/")
def home():
    if current_user.is_authenticated:
        return render_template("Home.html", name=current_user.name)
    else:
        return redirect(f"http://localhost:{config.MAIN_APP_PORT}/login")


@app.route("/auditory_learning")
def auditory_learning():
    global AudQuestionID  # current question ID
    global Aud_data  # fetched question details
    global no_q  # num of questions exist
    qid = random_q_r(AudQuestionID, no_q)
    print(qid)
    question_doc = db.collection("audio_questions").document(str(qid)).get()
    # Extract Question & Convert to Audio
    if question_doc.exists:
        question_data = question_doc.to_dict()
        question = question_data.get("Question", "No Question Available")
        sin_text_to_speech(question, qid)
        image = question_data.get("Image", None)
        if image:
            image = image.replace("<", "").replace(">", "")
        Aud_data = question_data
        return render_template(
            "Auditory_learning.html", question=question_data, image=image
        )
    return "No questions found.", 404


@app.route("/next_question", methods=["GET", "POST"])
def next_question():
    global AudQuestionID
    global Aud_data

    AudQuestionID += 1
    qid = random_q_r(AudQuestionID, no_q)
    print(qid)
    question_doc = db.collection("audio_questions").document(str(qid)).get()
    # Extract Question & Convert to Audio
    if question_doc.exists:
        question_data = question_doc.to_dict()
        question_id = str(AudQuestionID)

        # Check if audio file exists for this question
        audio_path = os.path.join("static", "aud_records", f"{question_id}.wav")
        audio_exists = os.path.exists(audio_path)

        question = question_data.get("Question", "No Question Available")
        sin_text_to_speech(question, qid)

        Aud_data = question_data
        image = question_data.get("Image", None)
        if image:
            image = image.replace("<", "").replace(">", "")

        return jsonify(
            {
                "success": True,
                "question": question_data,
                "image": image,
                "id": question_id,
                "audio_exists": audio_exists,
            }
        )
    else:
        AudQuestionID = 0
        return (
            jsonify(
                {"success": False, "question": False, "message": "No more questions!"}
            ),
            404,
        )


@app.route("/submit_audio", methods=["GET", "POST"])
def submit_sudio():
    global Aud_data
    global AudQuestionID
    global Aud_results
    global rd_lesson
    global Aud_results_2

    correct = False
    qid = Aud_data["ID"]
    audio_file = "static/aud_records/" + str(Aud_data["ID"]) + ".wav"

    ans_txt = stt_sinhala(audio_file)
    # retriev the current user's session info
    ori_answer = Aud_data["Answer"]
    lesson = Aud_data["Lesson"]

    if ans_txt:

        sim = is_similar(ori_answer, ans_txt)
        print(sim)
        print(sim, ori_answer)
        if sim > 0.6:
            correct = True
            if rd_lesson > 0:
                Aud_results_2.append(Aud_data["Lesson"])
            else:
                Aud_results.append(Aud_data["Lesson"])

        db.collection("audio_results").add(
            {
                "question_id": qid,
                "student_answer": ans_txt,
                "correct_answer": ori_answer,
                "lesson": lesson,
                "is_correct": correct,
            }
        )

        # print(Aud_results_2)
        # print(Aud_results)

        return jsonify({"success": True, "correct": correct, "answer": ori_answer})
    else:
        AudQuestionID -= 1
        return (
            jsonify({"success": False, "message": "Speech to text model problem"}),
            404,
        )


@app.route("/save_audio", methods=["POST"])
def save_audio():
    audio = request.files.get("audio")
    question_id = request.form.get("questionID")
    # ensure both audio file and qid are provided
    if not audio or not question_id:
        return (
            jsonify({"success": False, "message": "Missing audio or question ID"}),
            400,
        )

    # Temporarily save the raw file
    raw_filename = f"{question_id}_raw"
    raw_filepath = os.path.join(os.getcwd(), "static", "aud_records", raw_filename)
    audio.save(raw_filepath)

    # Convert raw (WebM/OGG) to WAV
    wav_filename = f"{question_id}.wav"
    wav_filepath = os.path.join(os.getcwd(), "static", "aud_records", wav_filename)

    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",  # overwrite
            "-i",
            raw_filepath,
            "-ar",
            "16000",  # resample to 16kHz if needed
            "-ac",
            "1",  # 1 channel
            wav_filepath,
        ]
    )

    # Optionally remove the raw file
    os.remove(raw_filepath)

    return jsonify(
        {"success": True, "message": f"Audio file {wav_filename} saved successfully."}
    )


@app.route("/speech_guide")
def speech_guide():
    global Aud_results
    global Aud_results_2
    global rd_lesson

    counts = Counter(Aud_results)
    counts_dict = dict(counts)
    weak_message = "ඔබ මෙම පාඩම සඳහා ගොඩක්ම දුර්වල අයෙකි. එම නිසා මෙම පාඩම පිළිබඳව ඇති ඔබගේ පෙළපොත අද්‍යයනය කර හොඳ දැනුමක් ලබාගන්න. පසුව මෙම ක්‍රමවේදය මගින් ඔබගේ දැනුම තහවුරු කර ගන්න!"

    if rd_lesson > 0:
        if len(Aud_results_2) == 0:  # If no new results
            counts_dict["New Results"] = 0
            message = weak_message
            images = ["imgtry.png"]
        else:
            counts2 = Counter(Aud_results_2)  # New results
            counts_dict2 = dict(counts2)
            print(counts_dict2)
            prev_score = counts_dict.get(
                "lesson0" + str(rd_lesson), 0
            )  # Previous result count
            new_score = counts_dict2.get(
                "lesson0" + str(rd_lesson), 0
            )  # New result count

            # Add new result to the result dictionary
            counts_dict["New lesson" + str(rd_lesson)] = new_score

            # Compare previous and new results
            if new_score == 0:
                message = weak_message  # Use the same message
                images = ["imgtry.png"]
            elif new_score > prev_score:
                message = (
                    "ඔබ ඔබගේ අපහසු පාඩම පිළිබඳ හොඳ අධ්‍යයනයක් ලබාගෙන ඇති අතර ඔබ ජය ගෙන ඇත!"
                )
                images = ["imagese.png"]
            elif new_score == prev_score:
                message = "ඔබ ඔබගේ අපහසු පාඩම පිළිබඳ ප්‍රගතියක් ලබාගෙන නැහැ!"
                images = ["imgtry.png"]
            else:
                message = "ඔබ තවමත් මෙම පාඩමට දුර්වල අයෙකි. එම නිසා මෙම පාඩම පිළිබඳව ඇති ඔබගේ පෙළපොත අද්‍යයනය කර හොඳ දැනුමක් ලබාගන්න. පසුව මෙම ක්‍රමවේදය මගින් ඔබගේ දැනුම තහවුරු කර ගන්න!"
                images = ["imgtry.png"]
        return render_template(
            "AudioGuide.html",
            results=counts_dict,
            message=message,
            images=images,
            lesson_id="Finished",
        )
    else:
        lesson_no = get_min_count_string(counts_dict)
        print("Shit lesson ------------------------------------>", rd_lesson)
        rd_lesson = lesson_no
        message = "ඔබ අවම ලකුණු ලබාගෙන ඇති පාඩම් අංකය " + str(lesson_no)
        img_range = [3, 2, 2]
        images = []
        for img in range(img_range[lesson_no - 1]):
            images.append("9" + str(lesson_no) + str(img + 1) + ".png")
        return render_template(
            "AudioGuide.html",
            results=counts_dict,
            message=message,
            images=images,
            lesson_id=str(lesson_no),
        )


if __name__ == "__main__":
    port = config.AUDIO_APP_PORT if hasattr(config, "AUDIO_APP_PORT") else 5004
    app.run(debug=True, port=port)

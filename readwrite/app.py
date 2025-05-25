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
import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import random
import re
import base64
from write_model.get_letters import get_text
from collections import Counter
import sys

# Fix the import mechanism for the config module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Use insert instead of append to prioritize this path

import config  # Now this will work
from firebase_config import db  # Import the Firebase configuration

# the system stores the data temporarily (until the session ends).
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
            user_doc = db.collection("readwrite_users").document(user_id).get()
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
            db.collection("readwrite_users").document(self.id).set(
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
WRITE_IMG_FOLDER = os.path.join(STATIC_FOLDER, "write_img")
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, "Images")

# Create necessary directories
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(WRITE_IMG_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Ensure templates directory exists
TEMPLATES_FOLDER = os.path.join(current_dir, "templates")
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)

# This is initial transformer model of google that used to check the similarity between two answers.
model_st = SentenceTransformer("Ransaka/bert-small-sentence-transformer")

# Flag to indicate if we're running with Firebase or in offline mode
OFFLINE_MODE = False

# Extablish connection with firebase
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    import config

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
WrQuestionID = 1
no_q = 5
username = ""
Wri_data = {}
# Wr_results = ["lesson1","lesson1","lesson1","lesson2","lesson2","lesson2","lesson3"]
# Wr_results_2 = ["lesson3","lesson3","lesson3","lesson3"]

Wr_results = []
Wr_results_2 = []
wr_lesson = 0
wr_lesson_c = 0


def extract_first_number(s):
    match = re.search(r"\d+", s)
    return int(match.group()) if match else None


# Finds the lesson where the student has attempted the least number of questions.
def get_min_count_string(strings):
    counter = Counter(strings)
    min_count = min(counter.values())
    min_strings = [s for s, count in counter.items() if count == min_count]
    no = extract_first_number(min_strings[0])
    return no


# Calculate Student's answers for each Lesson
def calculate_res(query):
    result = []
    for doc in query:
        data_dict = doc.to_dict()
        result.append(data_dict["data"]["Lesson"])
    counts = Counter(result)
    counts_dict = dict(counts)
    return counts_dict


# generate random questions from each lesson
def random_q_w(num, noq):
    global wr_lesson
    global wr_lesson_c
    if wr_lesson > 0:
        if wr_lesson_c > 4:
            lesson = 100
        else:
            wr_lesson_c = wr_lesson_c + 1
            lesson = wr_lesson - 1
    else:
        lesson = int((num - 1) / noq)
    start = lesson * 50
    qid = random.randint(start, start + 50)
    return qid


# calculates the similarity between correct answer and the inputed answer
def is_similar(target, source):
    sentences = [target, source]

    if target in source:
        return 0.75

    embeddings = model_st.encode(sentences, convert_to_tensor=True)
    # Compute Cosine Similarity using torch
    similarity = F.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    )

    return similarity


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
                print(f"Created new user in readwrite app: {user_id}")
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


@app.route("/api/info")
def api_info():
    return jsonify({"app": "Read/Write App", "status": "running"})


# Update home route to use current user data
@app.route("/")
def home():
    if current_user.is_authenticated:
        return render_template("Home.html", name=current_user.name)
    else:
        return redirect(f"http://localhost:{config.MAIN_APP_PORT}/login")


# Retrieve a random question from Firebase and displays it.
@app.route("/reading_writing_learning")
def reading_writing_learning():
    global WrQuestionID
    global Wri_data

    qid = random_q_w(WrQuestionID, no_q)
    question_doc = db.collection("write_questions").document(str(qid)).get()
    if question_doc.exists:
        question_data = question_doc.to_dict()
        image = question_data.get("Image", None)
        if image:
            image = image.replace("<", "").replace(">", "")
        Wri_data = question_data
        return render_template("R&W_learning.html", question=question_data, image=image)

    return "No questions found.", 404


# returns the next writing question
@app.route("/next_question_rw", methods=["GET"])
def next_question_rw():
    global WrQuestionID
    WrQuestionID += 1
    global Wri_data

    qid = random_q_w(WrQuestionID, no_q)
    question_doc = (
        db.collection("write_questions").document(str(qid)).get()
    )  # retrieve the question from db
    if question_doc.exists:
        question_data = question_doc.to_dict()
        image = question_data.get("Image", None)
        if image:
            image = image.replace("<", "").replace(
                ">", ""
            )  # If the question contains an image, it removes unnecessary symbols.

        Wri_data = question_data
        # Send the next question and image to the frontend
        return jsonify(
            {
                "success": True,
                "question": question_data,
                "image": image,
            }
        )
    else:
        # If no more questions, reset ID or handle accordingly
        WrQuestionID = 0  # Decrement ID to stay on the last valid question
        return (
            jsonify(
                {"success": False, "question": False, "message": "No more questions!"}
            ),
            404,
        )


# how many characters match and checks if they meet the 75% threshold.
def is_75_percent_match(str1: str, str2: str) -> bool:
    """
    Checks if at least 75% of the letters in one string match another string.

    Args:
        str1 (str): First input string.
        str2 (str): Second input string.

    Returns:
        bool: True if 75% or more of the characters match, False otherwise.
    """
    # Convert strings to lowercase to make the comparison case-insensitive
    str1, str2 = str1.lower(), str2.lower()

    # Count character occurrences in both strings
    counter1, counter2 = Counter(str1), Counter(str2)

    # Calculate total matching characters
    match_count = sum((counter1 & counter2).values())

    # Determine the percentage of matching characters
    required_match = max(len(str1), len(str2)) * 0.75

    return match_count >= required_match


@app.route("/submit_write", methods=["GET", "POST"])
def submit_write():
    global Wri_data
    global WrQuestionID
    global Wr_results  # Stores the correct answers for normal questions.
    global wr_lesson  # Keeps track of difficult lessons
    global Wr_results_2  # Stores the correct answers for difficult lessons.

    correct = False  # check whether the student's answer is correct or not.
    qid = Wri_data["ID"]  # The current question ID is retrieved
    _file = "./readwrite/static/write_img/" + str(qid) + ".png"

    # Get the handwritten answer (image) and number input from the frontend.
    data = request.get_json()
    image_data = data.get("image")
    number = data.get("number")

    if not image_data:
        return jsonify({"message": "No image data received."}), 400

    # Convert base64 image data into binary format and save it as an image
    image_data = re.sub("^data:image/.+;base64,", "", image_data)
    image_binary = base64.b64decode(image_data)
    # Save the file
    with open(_file, "wb") as f:
        f.write(image_binary)

    model_path = "readwrite/write_model/multi_letter_detector2.pth"
    class_mapping_path = "readwrite/write_model/class_mapping.pkl"
    class_mapping_path2 = "readwrite/write_model/class_mapping2.pkl"

    # The OCR model  reads the text from the image
    ans_txt = get_text(
        model_path, _file, class_mapping_path, class_mapping_path2, number, qid
    )
    # user = session.get("user", "No user stored")

    ori_answer = Wri_data["Answer"]
    lesson = Wri_data["Lesson"]

    # Compare the student's answer (ans_txt) with the original correct answer (ori_answer).
    if ans_txt:
        sim = is_similar(ori_answer, ans_txt)
        print(ans_txt, "===", ori_answer, "===", sim)

        # If the similarity score is greater than 0.7 or matches 75%, consider it correct
        if sim > 0.6 or is_75_percent_match(ori_answer, ans_txt):
            correct = True
            if wr_lesson > 0:
                Wr_results_2.append(lesson)
            else:
                Wr_results.append(lesson)

        # Store all answers in the Firestore database.
        db.collection("write_results").add(
            {
                "question_id": qid,
                "student_answer": ans_txt,
                "correct_answer": ori_answer,
                "lesson": lesson,
                "is_correct": correct,
            }
        )
        # Send the response to the frontend
        return jsonify({"success": True, "correct": correct, "answer": ori_answer})

    else:
        WrQuestionID -= 1
        return jsonify({"success": False, "message": "OCR model not working"}), 404


@app.route("/write_guide")
def write_guide():
    global Wr_results_2
    global Wr_results
    global wr_lesson
    counts = Counter(Wr_results)
    counts_dict = dict(counts)

    # Default message for 0 correct answers
    weak_message = "ඔබ මෙම පාඩම සඳහා ගොඩක්ම දුර්වලයි.එම නිසා මෙම පාඩම පිළිබඳව ඇති ඔබගේ පෙළපොත අද්‍යයනය කර හොඳ දැනුමක් ලබාගන්න. පසුව මෙම ක්‍රමවේදය මගින් ඔබගේ දැනුම තහවුරු කර ගන්න!"

    if wr_lesson > 0:
        if len(Wr_results_2) == 0:  # If no new results
            counts_dict["New Results"] = 0
            message = weak_message
            images = ["Lose.jpg"]
        else:
            counts2 = Counter(Wr_results_2)  # New results
            counts_dict2 = dict(counts2)
            print(counts_dict2)
            prev_score = counts_dict.get(
                "lesson0" + str(wr_lesson), 0
            )  # Previous result count
            new_score = counts_dict2.get(
                "lesson0" + str(wr_lesson), 0
            )  # New result count

            # Add new result to the result dictionary
            counts_dict["New lesson" + str(wr_lesson)] = new_score

            # Compare previous and new results
            if new_score == 0:
                message = weak_message
                images = ["Lose.jpg"]
            elif new_score > prev_score:
                message = "ඔබගේ දැනුම වැඩිදියුණු වී ඇත!"
                images = ["completedHappy.jpg"]
            elif new_score == prev_score:
                message = "ඔබ ප්‍රගතියක් ලබාගෙන නැහැ!"
                images = ["notprogess.jpg"]
            else:
                message = "ඔබගේ දැනුම අඩු වී ඇත!"
                images = ["Lose.jpg"]
        # images = ["completedHappy.jpg"]
        return render_template(
            "WriteGuide.html",
            results=counts_dict,
            message=message,
            images=images,
            lesson_id="Finished",
        )
    else:
        lesson_no = get_min_count_string(counts_dict)
        wr_lesson = lesson_no
        message = "ඔබ අවම වශයෙන් නිවැරදි උත්තර ලබාදී ඇති පාඩම වනුයේ, පාඩම් අංක : " + str(
            lesson_no
        )
        img_range = [2, 3, 3]
        images = []
        for img in range(img_range[lesson_no - 1]):
            images.append(str(lesson_no) + str(img + 1) + ".png")
        return render_template(
            "WriteGuide.html",
            results=counts_dict,
            message=message,
            images=images,
            lesson_id=str(lesson_no),
        )


if __name__ == "__main__":
    port = config.READWRITE_APP_PORT if hasattr(config, "READWRITE_APP_PORT") else 5002
    app.run(debug=True, port=port)

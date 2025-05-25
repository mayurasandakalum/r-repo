import json
import os
from firebase_admin import firestore
from .models import Question, SubQuestion


def load_initial_questions():
    db = firestore.client()

    # Check if questions already exist
    existing_questions = db.collection("questions").limit(1).get()
    if len(list(existing_questions)) > 0:
        print("Questions already exist in Firebase")
        return

    # Load questions from JSON
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "initial_questions.json")

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for q_data in data["questions"]:
            # Create question document
            question_ref = db.collection("questions").document()
            question_ref.set(
                {
                    "text": q_data["text"],
                    "answer_method": q_data.get("answer_method", "abacus"),
                    "is_published": q_data.get("is_published", True),
                    "created": firestore.SERVER_TIMESTAMP,
                    "modified": firestore.SERVER_TIMESTAMP,
                }
            )

            # Create sub-questions for this question
            for sub_q in q_data["sub_questions"]:
                sub_q_ref = db.collection("sub_questions").document()
                sub_q_ref.set(
                    {
                        "question_id": question_ref.id,
                        "text": sub_q["text"],
                        "instructions": sub_q.get("instructions", ""),
                        "correct_answer": sub_q["correct_answer"],
                        "answer_type": sub_q.get("answer_type", "number"),
                        "min_value": sub_q.get("min_value"),
                        "max_value": sub_q.get("max_value"),
                        "time_format": sub_q.get("time_format"),
                        "difficulty_level": sub_q.get("difficulty_level", 1),
                        "points": sub_q.get("points", 1),
                        "hint": sub_q.get("hint"),
                        "created": firestore.SERVER_TIMESTAMP,
                        "modified": firestore.SERVER_TIMESTAMP,
                    }
                )

        print("Successfully loaded initial questions into Firebase")
    except Exception as e:
        print(f"Error loading initial questions: {str(e)}")

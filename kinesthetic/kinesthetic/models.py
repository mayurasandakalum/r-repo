from flask_login import UserMixin
from datetime import datetime
import random
from firebase_admin import firestore
import firebase_admin
from firebase_admin import credentials
import os
import uuid  # Add this import

# Get the absolute path to the credentials file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Remove one dirname call
cred_path = os.path.join(project_root, "serviceAccountKey.json")

print(f"Looking for credentials at: {cred_path}")  # Debug print

# Initialize Firebase
cred = credentials.Certificate(cred_path)
try:
    firebase_admin.initialize_app(cred)
except ValueError:
    # App already initialized
    pass
db = firestore.client()


class User(UserMixin):
    def __init__(self, username, email, first_name, last_name, password_hash, id=None):
        self.id = id if id else str(uuid.uuid4())
        self.username = username
        self.email = email
        self.first_name = first_name
        self.last_name = last_name
        self.password_hash = password_hash
        self.created = datetime.utcnow()
        self.modified = datetime.utcnow()

    @staticmethod
    def get_by_id(user_id):
        user_doc = db.collection("users").document(str(user_id)).get()
        if user_doc.exists:
            data = user_doc.to_dict()
            return User(
                username=data.get("username"),
                email=data.get("email"),
                first_name=data.get("first_name"),
                last_name=data.get("last_name"),
                password_hash=data.get("password_hash"),
                id=user_doc.id,
            )
        return None

    @staticmethod
    def get_by_username(username):
        users = db.collection("users").where("username", "==", username).limit(1).get()
        for user in users:
            data = user.to_dict()
            return User(
                username=data.get("username"),
                email=data.get("email"),
                first_name=data.get("first_name"),
                last_name=data.get("last_name"),
                password_hash=data.get("password_hash"),
                id=user.id,
            )
        return None

    def save(self):
        data = {
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "password_hash": self.password_hash,
            "created": self.created,
            "modified": self.modified,
        }
        db.collection("users").document(str(self.id)).set(data)


class QuizProfile:
    def __init__(self, user_id, total_score=0.0, created=None, modified=None, 
                 completed_lessons=None, current_lesson_attempts=0, 
                 mixed_quiz_completed=False, subject_counts=None,
                 subject_performance=None, watched_videos=None,
                 quiz_comparisons=None, weakest_subject=None,
                 weakest_subject_performance=None):  # Add new parameter
        self.user_id = user_id
        self.total_score = total_score
        self.created = created if created else datetime.utcnow()
        self.modified = modified if modified else datetime.utcnow()
        self._user = None  # Cache for user object
        self.completed_lessons = completed_lessons or []
        self.current_lesson_attempts = current_lesson_attempts
        self.mixed_quiz_completed = mixed_quiz_completed
        self.subject_counts = subject_counts or {}  # To track how many questions from each subject have been shown
        # Track performance by subject for first quiz: {subject: {"correct": x, "total": y, "score": z}}
        self.subject_performance = subject_performance or {}
        # Track performance by subject for second quiz (weakest subject): {subject: {"correct": x, "total": y, "score": z}}
        self.weakest_subject_performance = weakest_subject_performance or {}
        # Track which subject videos have been watched
        self.watched_videos = watched_videos or []
        # Store quiz comparison data: {subject: {"before": {"correct": a, "total": b, "percentage": c}, 
        #                                      "after": {"correct": d, "total": e, "percentage": f}}}
        self.quiz_comparisons = quiz_comparisons or {}
        # Store the weakest subject after first quiz
        self.weakest_subject = weakest_subject

    @staticmethod
    def get_by_user_id(user_id):
        profile = db.collection("kinesthetic_profiles").document(str(user_id)).get()
        if profile.exists:
            data = profile.to_dict()
            profile = QuizProfile(
                user_id=data.get("user_id"),
                total_score=data.get("total_score", 0.0),
                created=data.get("created"),
                modified=data.get("modified"),
                completed_lessons=data.get("completed_lessons", []),
                current_lesson_attempts=data.get("current_lesson_attempts", 0),
                mixed_quiz_completed=data.get("mixed_quiz_completed", False),
                subject_counts=data.get("subject_counts", {}),
                subject_performance=data.get("subject_performance", {}),
                watched_videos=data.get("watched_videos", []),
                quiz_comparisons=data.get("quiz_comparisons", {}),
                weakest_subject=data.get("weakest_subject"),
                weakest_subject_performance=data.get("weakest_subject_performance", {})  # Add this line
            )
            return profile
        return None

    def save(self):
        data = {
            "user_id": self.user_id,
            "total_score": self.total_score,
            "created": self.created,
            "modified": self.modified,
            "completed_lessons": self.completed_lessons,
            "current_lesson_attempts": self.current_lesson_attempts,
            "mixed_quiz_completed": self.mixed_quiz_completed,
            "subject_counts": self.subject_counts,
            "subject_performance": self.subject_performance,
            "watched_videos": self.watched_videos,
            "quiz_comparisons": self.quiz_comparisons,
            "weakest_subject": self.weakest_subject,
            "weakest_subject_performance": self.weakest_subject_performance,  # Add this line
        }
        db.collection("kinesthetic_profiles").document(str(self.user_id)).set(data)

    def get_new_question(self):
        # Get all attempted questions
        attempts = (
            db.collection("attempted_questions")
            .where("user_id", "==", self.user_id)
            .get()
        )
        attempted_ids = [attempt.get("question_id") for attempt in attempts]

        # Get remaining questions
        questions = db.collection("questions").where("is_published", "==", True).get()
        available_questions = [q for q in questions if q.id not in attempted_ids]

        if available_questions:
            question = random.choice(available_questions)
            return Question.from_doc(question)
        return None

    def evaluate_attempt(self, attempted_question):
        # Get the selected choice
        choice_ref = (
            db.collection("choices")
            .document(attempted_question.selected_choice_id)
            .get()
        )
        if not choice_ref.exists:
            return False

        choice_data = choice_ref.to_dict()
        is_correct = choice_data.get("is_correct", False)

        if is_correct:
            self.total_score += 1
            self.modified = datetime.utcnow()
            self.save()

        return is_correct

    def get_user(self):
        if self._user is None:
            self._user = User.get_by_id(self.user_id)
        return self._user

    @property
    def user(self):
        return self.get_user()

    def get_weakest_subject(self):
        """Returns the subject with the lowest performance percentage"""
        weakest = None
        lowest_percentage = 100  # Start with maximum possible percentage
        
        for subject, data in self.subject_performance.items():
            correct = data.get("correct", 0)
            total = data.get("total", 0)
            
            if total == 0:  # Skip subjects with no attempts
                continue
                
            percentage = (correct / total) * 100
            if percentage < lowest_percentage:
                lowest_percentage = percentage
                weakest = subject
            # If percentages are equal, compare the raw scores
            elif percentage == lowest_percentage and weakest is not None:
                # If the current subject has a lower score, it becomes the weakest
                current_score = data.get("score", 0)
                weakest_score = self.subject_performance[weakest].get("score", 0)
                if current_score < weakest_score:
                    weakest = subject
        
        return {
            "subject": weakest,
            "percentage": lowest_percentage if weakest else 0
        }


class AnswerMethod:
    ABACUS = "abacus"
    ANALOG_CLOCK = "analog_clock"
    DIGITAL_CLOCK = "digital_clock"

    CHOICES = [
        (ABACUS, "Abacus"),
        (ANALOG_CLOCK, "Analog Clock"),
        (DIGITAL_CLOCK, "Digital Clock"),
    ]


class Subject:
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    TIME = "time"

    CHOICES = [
        (ADDITION, "එකතු කිරීම පාඩම"),
        (SUBTRACTION, "අඩු කිරීම පාඩම"),
        (TIME, "කාලය පාඩම"),
    ]

    ANSWER_METHODS = {
        ADDITION: [(AnswerMethod.ABACUS, "Abacus")],
        SUBTRACTION: [(AnswerMethod.ABACUS, "Abacus")],
        TIME: [
            (AnswerMethod.ANALOG_CLOCK, "Analog Clock"),
            (AnswerMethod.DIGITAL_CLOCK, "Digital Clock"),
        ],
    }


class Question:
    def __init__(
        self,
        id=None,
        text="",
        subject=Subject.ADDITION,  # Default subject
        answer_method=AnswerMethod.ABACUS,
        is_published=False,
    ):
        self.id = id
        self.text = text
        self.subject = subject
        self.answer_method = answer_method
        self.is_published = is_published
        self.created = datetime.utcnow()
        self.modified = datetime.utcnow()
        self._sub_questions = None  # Cache for sub-questions

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()
        # Ensure subject is always set, default to ADDITION if not present
        subject = data.get("subject")
        if not subject:
            subject = Subject.ADDITION
            # Update the document with the default subject
            db.collection("questions").document(doc.id).update({"subject": subject})

        question = Question(
            id=doc.id,
            text=data.get("text", ""),
            subject=subject,
            answer_method=data.get("answer_method", AnswerMethod.ABACUS),
            is_published=data.get("is_published", False),
        )
        question.created = data.get("created", datetime.utcnow())
        question.modified = data.get("modified", datetime.utcnow())
        return question

    def save(self):
        data = {
            "text": self.text,
            "subject": self.subject,
            "answer_method": self.answer_method,
            "is_published": self.is_published,
            "modified": datetime.utcnow(),
        }

        if not self.id:
            # New question
            data["created"] = datetime.utcnow()
            ref = db.collection("questions").add(data)
            self.id = ref[1].id
        else:
            # Update existing question
            db.collection("questions").document(self.id).update(data)

    @property
    def sub_questions(self):
        # Return cached sub-questions if available
        if self._sub_questions is not None:
            return self._sub_questions

        # Otherwise load from database
        self._sub_questions = SubQuestion.get_by_question(self.id)
        return self._sub_questions


class SubQuestion:
    def __init__(
        self,
        id=None,
        question_id=None,
        text="",  # The sub-question text
        instructions="",  # Instructions for answering
        correct_answer="",  # Expected answer
        answer_type="number",  # Type of answer expected (number, time, etc)
        min_value=None,  # For numeric answers: minimum allowed value
        max_value=None,  # For numeric answers: maximum allowed value
        time_format=None,  # For time answers: format specification
        difficulty_level=1,  # Difficulty level 1-5
        points=1,  # Points awarded for correct answer
        hint=None,  # Optional hint for the student
    ):
        self.id = id
        self.question_id = question_id
        self.text = text
        self.instructions = instructions
        self.correct_answer = correct_answer
        self.answer_type = answer_type
        self.min_value = min_value
        self.max_value = max_value
        self.time_format = time_format
        self.difficulty_level = difficulty_level
        self.points = points
        self.hint = hint
        self.created = datetime.utcnow()
        self.modified = datetime.utcnow()

    @staticmethod
    def get_by_question(question_id):
        # Add index hint for better performance
        subquestions = (
            db.collection("sub_questions")
            .where("question_id", "==", question_id)
            .order_by("created", direction=firestore.Query.ASCENDING)
            .get()
        )
        return [SubQuestion.from_doc(doc) for doc in subquestions]

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()
        subq = SubQuestion(
            id=doc.id,
            question_id=data.get("question_id"),
            text=data.get("text", ""),
            instructions=data.get("instructions", ""),
            correct_answer=data.get("correct_answer", ""),
            answer_type=data.get("answer_type", "number"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            time_format=data.get("time_format"),
            difficulty_level=data.get("difficulty_level", 1),
            points=data.get("points", 1),
            hint=data.get("hint"),
        )
        subq.created = data.get("created", datetime.utcnow())
        subq.modified = data.get("modified", datetime.utcnow())
        return subq

    def save(self):
        data = {
            "question_id": self.question_id,
            "text": self.text,
            "instructions": self.instructions,
            "correct_answer": self.correct_answer,
            "answer_type": self.answer_type,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "time_format": self.time_format,
            "difficulty_level": self.difficulty_level,
            "points": self.points,
            "hint": self.hint,
            "created": self.created,
            "modified": self.modified,
        }
        if self.id:
            db.collection("sub_questions").document(self.id).set(data)
        else:
            ref = db.collection("sub_questions").add(data)
            self.id = ref[1].id


class AttemptedQuestion:
    def __init__(
        self, user_id, question_id, sub_question_id=None, is_correct=False, images=None,
        result_data=None, quiz_type="mixed_quiz"  # Add quiz_type field with default value
    ):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.question_id = question_id
        self.sub_question_id = sub_question_id
        self.is_correct = is_correct
        self.images = images or {}
        self.result_data = result_data or {}
        self.quiz_type = quiz_type  # Initialize quiz_type
        self.attempted_at = datetime.utcnow()

    def save(self):
        data = {
            "user_id": self.user_id,
            "question_id": self.question_id,
            "sub_question_id": self.sub_question_id,
            "is_correct": self.is_correct,
            "images": self.images,
            "result_data": self.result_data,  # Add this field
            "quiz_type": self.quiz_type,  # Save quiz_type
            "attempted_at": self.attempted_at,
        }
        db.collection("attempted_questions").document(self.id).set(data)

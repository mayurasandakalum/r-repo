from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from firebase_admin import firestore
from datetime import datetime  # Add this import at the top
import random  # Add this import

from .models import (
    User,
    QuizProfile,
    Question,
    AttemptedQuestion,
    SubQuestion,
    Subject,
)
from .forms import (
    UserLoginForm,
    RegistrationForm,
    QuizForm,
    QuestionForm,
    SubQuestionForm,
)  # Added QuestionForm and SubQuestionForm

db = firestore.client()

quiz_blueprint = Blueprint(
    "quiz", __name__, template_folder="../templates/quiz", static_folder="../static"
)


@quiz_blueprint.route("/")
def home():
    return render_template("quiz/home.html")


@quiz_blueprint.route("/user-home")
@login_required
def user_home():
    quiz_profile = QuizProfile.get_by_user_id(current_user.id)
    return render_template("quiz/user_home.html", quiz_profile=quiz_profile)


@quiz_blueprint.route("/leaderboard")
def leaderboard():
    # Get all quiz profiles and sort by total_score
    profiles_ref = (
        db.collection("quiz_profiles")
        .order_by("total_score", direction=firestore.Query.DESCENDING)
        .limit(500)
    )
    profiles = profiles_ref.get()
    top_quiz_profiles = [QuizProfile(**profile.to_dict()) for profile in profiles]
    total_count = len(top_quiz_profiles)
    return render_template(
        "quiz/leaderboard.html",
        top_quiz_profiles=top_quiz_profiles,
        total_count=total_count,
    )


@quiz_blueprint.route("/play", methods=["GET", "POST"])
@login_required
def play():
    quiz_profile = QuizProfile.get_by_user_id(current_user.id)
    if not quiz_profile:
        quiz_profile = QuizProfile(user_id=current_user.id)
        quiz_profile.save()

    subject = request.args.get("subject", Subject.ADDITION)

    # Check if user has completed all lessons
    if len(quiz_profile.completed_lessons) == len(Subject.CHOICES):
        return render_template("quiz/all_lessons_completed.html")

    # Check if current subject is completed and redirect to next subject
    if subject in quiz_profile.completed_lessons:
        next_subject = None
        for s, _ in Subject.CHOICES:
            if s not in quiz_profile.completed_lessons:
                next_subject = s
                break
        if next_subject:
            return redirect(url_for("quiz.lesson_instructions", subject=next_subject))
        return redirect(url_for("quiz.leaderboard"))

    # Handle POST request for answering questions
    if request.method == "POST":
        question_id = request.form.get("question_pk")
        answer_method = request.form.get("answer_method")
        sub_question_id = request.form.get("sub_question_id")

        # Get all the captured images
        captured_images = {}
        for key in request.form:
            if key.startswith("captured_image_webcam"):
                captured_images[key] = request.form[key]

        # Get the sub-question to check correct answer
        sub_question_ref = (
            db.collection("sub_questions").document(sub_question_id).get()
        )
        if sub_question_ref.exists:
            sub_question_data = sub_question_ref.to_dict()
            correct_answer = sub_question_data.get("correct_answer")
            points = sub_question_data.get("points", 1)

            # TODO: Implement image processing logic here
            # is_correct = process_answer(answer_method, captured_images, correct_answer)
            is_correct = True  # Temporary, replace with actual logic

            # Save attempt
            attempted = AttemptedQuestion(
                user_id=current_user.id,
                question_id=question_id,
                sub_question_id=sub_question_id,
                is_correct=is_correct,
                images=captured_images,
            )
            attempted.save()

            # Update score if correct
            if is_correct:
                quiz_profile = QuizProfile.get_by_user_id(current_user.id)
                if quiz_profile:
                    quiz_profile.total_score += points
                    quiz_profile.save()

        # Update attempts counter
        quiz_profile.current_lesson_attempts += 1

        # Check if lesson is complete (5 questions answered)
        if quiz_profile.current_lesson_attempts >= 5:
            quiz_profile.completed_lessons.append(subject)
            quiz_profile.current_lesson_attempts = 0
            quiz_profile.save()
            if len(quiz_profile.completed_lessons) == len(Subject.CHOICES):
                return redirect(url_for("quiz.all_lessons_completed"))
            return redirect(url_for("quiz.user_home"))

        quiz_profile.save()
        return redirect(url_for("quiz.play", subject=subject))

    # Handle GET request
    # Get 5 random questions for the current subject
    questions_ref = (
        db.collection("questions")
        .where("subject", "==", subject)
        .where("is_published", "==", True)
        .get()
    )
    available_questions = [Question.from_doc(q) for q in questions_ref]

    if not available_questions:
        flash("No questions available for this subject.", "warning")
        return redirect(url_for("quiz.user_home"))

    # Add default value for remaining_questions
    remaining_questions = 5 - (quiz_profile.current_lesson_attempts or 0)

    if quiz_profile.current_lesson_attempts >= 5:
        quiz_profile.completed_lessons.append(subject)
        quiz_profile.current_lesson_attempts = 0
        quiz_profile.save()

        # Find next subject
        next_subject = None
        for s, _ in Subject.CHOICES:
            if s not in quiz_profile.completed_lessons:
                next_subject = s
                break

        if next_subject:
            return redirect(url_for("quiz.lesson_instructions", subject=next_subject))
        return redirect(url_for("quiz.leaderboard"))

    # Select a random question
    question = random.choice(available_questions)

    return render_template(
        "quiz/play.html",
        question=question,
        subject=subject,
        remaining_questions=remaining_questions,
    )


@quiz_blueprint.route("/submission-result/<int:attempted_question_pk>")
@login_required
def submission_result(attempted_question_pk):
    attempted_question = AttemptedQuestion.query.get_or_404(attempted_question_pk)
    return render_template(
        "quiz/submission_result.html", attempted_question=attempted_question
    )


@quiz_blueprint.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("quiz.home"))

    form = UserLoginForm()
    if form.validate_on_submit():
        user = User.get_by_username(form.username.data)
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            return redirect(url_for("quiz.user_home"))
        else:
            flash("Invalid username/password!", "danger")
    return render_template("quiz/login.html", form=form, title="Login")


@quiz_blueprint.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("quiz.home"))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User.get_by_username(form.username.data)
        if user:
            flash("Username already exists")
            return redirect(url_for("quiz.register"))

        password_hash = generate_password_hash(form.password.data)
        new_user = User(
            username=form.username.data,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            password_hash=password_hash,
        )

        new_user.save()

        quiz_profile = QuizProfile(user_id=new_user.id)
        quiz_profile.save()

        flash("Registration successful!")
        return redirect(url_for("quiz.login"))

    return render_template("quiz/registration.html", form=form, title="Register")


@quiz_blueprint.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("quiz.home"))


def batch_get_subquestions(question_ids):
    """Helper function to get sub-questions in batches"""
    sub_questions_by_question = {}

    # Process question_ids in batches of 30
    for i in range(0, len(question_ids), 30):
        batch_ids = question_ids[i : i + 30]
        sub_questions_ref = (
            db.collection("sub_questions").where("question_id", "in", batch_ids).get()
        )

        # Group sub-questions by question_id
        for sub_doc in sub_questions_ref:
            sub_q = SubQuestion.from_doc(sub_doc)
            if sub_q.question_id not in sub_questions_by_question:
                sub_questions_by_question[sub_q.question_id] = []
            sub_questions_by_question[sub_q.question_id].append(sub_q)

    return sub_questions_by_question


@quiz_blueprint.route("/manage/questions")
def manage_questions():
    # Get all questions in a single query
    questions_ref = (
        db.collection("questions")
        .order_by("created", direction=firestore.Query.DESCENDING)
        .get()
    )

    # Get all sub-questions in batches
    all_questions = [Question.from_doc(doc) for doc in questions_ref]
    question_ids = [q.id for q in all_questions]

    # Get sub-questions using batching
    sub_questions_by_question = batch_get_subquestions(question_ids)

    # Organize questions by subject
    questions_by_subject = {}
    for question in all_questions:
        if question.subject not in questions_by_subject:
            questions_by_subject[question.subject] = []
        question._sub_questions = sub_questions_by_question.get(question.id, [])
        questions_by_subject[question.subject].append(question)

    return render_template(
        "quiz/manage/questions_list.html",
        questions_by_subject=questions_by_subject,
        subjects=Subject.CHOICES,
    )


@quiz_blueprint.route("/manage/questions/new", methods=["GET", "POST"])
def new_question():
    # Get subject from query parameter if it exists
    subject = request.args.get("subject", Subject.ADDITION)
    form = QuestionForm(initial_subject=subject)

    if form.validate_on_submit():
        question = Question(
            text=form.text.data,
            subject=form.subject.data,  # Make sure to save the subject
            answer_method=form.answer_method.data,
            is_published=form.is_published.data,
        )
        question.save()

        # Create sub-questions
        for sub_form in form.sub_questions:
            subquestion = SubQuestion(
                question_id=question.id,
                text=sub_form.text.data,
                instructions=sub_form.instructions.data,
                correct_answer=sub_form.correct_answer.data,
                answer_type=sub_form.answer_type.data,
                min_value=sub_form.min_value.data,
                max_value=sub_form.max_value.data,
                time_format=sub_form.time_format.data,
                difficulty_level=sub_form.difficulty_level.data,
                points=sub_form.points.data,
                hint=sub_form.hint.data,
            )
            subquestion.save()

        flash("Question and sub-questions created successfully!", "success")
        return redirect(url_for("quiz.manage_questions"))
    return render_template(
        "quiz/manage/question_form.html",
        form=form,
        title="New Question",
        initial_subject=subject,
    )


@quiz_blueprint.route("/manage/questions/<question_id>", methods=["GET", "POST"])
@login_required  # Keep this decorator here
def edit_question(question_id):
    question_ref = db.collection("questions").document(question_id).get()
    if not question_ref.exists:
        flash("Question not found!", "error")
        return redirect(url_for("quiz.manage_questions"))

    question = Question.from_doc(question_ref)
    form = QuestionForm(obj=question)

    if form.validate_on_submit():
        # Update question fields
        question.text = form.text.data
        question.subject = form.subject.data
        question.answer_method = form.answer_method.data
        question.is_published = form.is_published.data
        question.modified = datetime.utcnow()  # Now datetime is properly imported

        try:
            # Update the question in the database
            question.save()
            flash("Question updated successfully!", "success")
            return redirect(url_for("quiz.manage_questions"))
        except Exception as e:
            flash(f"Error updating question: {str(e)}", "error")
            return redirect(url_for("quiz.edit_question", question_id=question_id))

    return render_template(
        "quiz/manage/question_form.html",
        form=form,
        question=question,
        title="Edit Question",
    )


@quiz_blueprint.route(
    "/manage/questions/<question_id>/subquestions/new", methods=["GET", "POST"]
)
def new_subquestion(question_id):
    form = SubQuestionForm()
    if form.validate_on_submit():
        subquestion = SubQuestion(
            question_id=question_id,
            text=form.text.data,
            instructions=form.instructions.data,
            correct_answer=form.correct_answer.data,
            answer_type=form.answer_type.data,
            min_value=form.min_value.data,
            max_value=form.max_value.data,
            time_format=form.time_format.data,
            difficulty_level=form.difficulty_level.data,
            points=form.points.data,
            hint=form.hint.data,
        )
        subquestion.save()
        flash("Sub-question added successfully!", "success")
        return redirect(url_for("quiz.manage_questions"))  # Changed this line
    return render_template(
        "quiz/manage/subquestion_form.html",
        form=form,
        question_id=question_id,
        title="New Sub-question",
    )


@quiz_blueprint.route("/manage/subquestions/<subquestion_id>", methods=["GET", "POST"])
def edit_subquestion(subquestion_id):
    subquestion_ref = db.collection("sub_questions").document(subquestion_id).get()
    if not subquestion_ref.exists:
        flash("Sub-question not found!", "error")
        return redirect(url_for("quiz.manage_questions"))

    subquestion = SubQuestion.from_doc(subquestion_ref)
    form = SubQuestionForm(obj=subquestion)

    if form.validate_on_submit():
        subquestion.text = form.text.data
        subquestion.instructions = form.instructions.data
        subquestion.correct_answer = form.correct_answer.data
        subquestion.answer_type = form.answer_type.data
        subquestion.min_value = form.min_value.data
        subquestion.max_value = form.max_value.data
        subquestion.time_format = form.time_format.data
        subquestion.difficulty_level = form.difficulty_level.data
        subquestion.points = form.points.data
        subquestion.hint = form.hint.data
        subquestion.save()
        flash("Sub-question updated successfully!", "success")
        return redirect(url_for("quiz.manage_questions"))  # Changed this line

    return render_template(
        "quiz/manage/subquestion_form.html",
        form=form,
        subquestion=subquestion,
        title="Edit Sub-question",
    )


@quiz_blueprint.route("/api/answer-methods/<subject>")
def get_answer_methods(subject):  # Remove @login_required decorator
    # Get the answer methods for the selected subject
    methods = Subject.ANSWER_METHODS.get(subject, [])
    return jsonify({"methods": methods})


@quiz_blueprint.route("/manage/questions/<question_id>/delete", methods=["POST"])
@login_required
def delete_question(question_id):
    try:
        # Delete all sub-questions first
        sub_questions = (
            db.collection("sub_questions").where("question_id", "==", question_id).get()
        )
        for sub_q in sub_questions:
            sub_q.reference.delete()

        # Then delete the question
        db.collection("questions").document(question_id).delete()
        flash("Question deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting question: {str(e)}", "error")

    return redirect(url_for("quiz.manage_questions"))


@quiz_blueprint.route("/manage/subquestions/<subquestion_id>/delete", methods=["POST"])
@login_required
def delete_subquestion(subquestion_id):
    try:
        db.collection("sub_questions").document(subquestion_id).delete()
        flash("Sub-question deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting sub-question: {str(e)}", "error")

    return redirect(url_for("quiz.manage_questions"))


@quiz_blueprint.route("/lesson-instructions/<subject>")
@login_required
def lesson_instructions(subject):
    subject_names = {
        "addition": "එකතු කිරීම පාඩම",
        "subtraction": "අඩු කිරීම පාඩම",
        "time": "කාලය පාඩම",
    }

    if subject not in subject_names:
        flash("Invalid subject selected", "error")
        return redirect(url_for("quiz.user_home"))

    return render_template(
        "quiz/lesson_instructions.html",
        subject=subject,
        subject_name=subject_names[subject],
    )

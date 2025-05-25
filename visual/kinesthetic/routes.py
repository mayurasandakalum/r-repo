from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify, current_app
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from firebase_admin import firestore
from datetime import datetime
import random
import base64
import os
import shutil

from .models import (
    User,
    QuizProfile,
    Question,
    AttemptedQuestion,
    SubQuestion,
    Subject,
    AnswerMethod,
)
from .forms import (
    UserLoginForm,
    RegistrationForm,
    QuizForm,
    QuestionForm,
    SubQuestionForm,
)
from services.abacus_service import check_abacus_answer
from services.clock_service import check_clock_answer  # Add this import

db = firestore.client()

kinesthetic_blueprint = Blueprint(
    "kinesthetic",
    __name__,
    template_folder="../templates/kinesthetic",
    static_folder="../static",
)


@kinesthetic_blueprint.route("/")
def home():
    return render_template("kinesthetic/home.html")


@kinesthetic_blueprint.route("/user-home")
@login_required
def user_home():
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    return render_template(
        "kinesthetic/user_home.html", kinesthetic_profile=kinesthetic_profile
    )


@kinesthetic_blueprint.route("/leaderboard")
def leaderboard():
    # Get all kinesthetic profiles and sort by total_score
    profiles_ref = (
        db.collection("kinesthetic_profiles")
        .order_by("total_score", direction=firestore.Query.DESCENDING)
        .limit(500)
    )
    profiles = profiles_ref.get()
    top_kinesthetic_profiles = [
        QuizProfile(**profile.to_dict()) for profile in profiles
    ]
    total_count = len(top_kinesthetic_profiles)
    return render_template(
        "kinesthetic/leaderboard.html",
        top_kinesthetic_profiles=top_kinesthetic_profiles,
        total_count=total_count,
    )


@kinesthetic_blueprint.route("/play", methods=["GET", "POST"])
@login_required
def play():
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    if not kinesthetic_profile:
        kinesthetic_profile = QuizProfile(user_id=current_user.id)
        kinesthetic_profile.save()

    # Check if user has completed all 15 questions
    if kinesthetic_profile.mixed_quiz_completed:
        return render_template("kinesthetic/all_lessons_completed.html")

    # Handle POST request for answering questions
    if request.method == "POST":
        question_id = request.form.get("question_pk")
        answer_method = request.form.get("answer_method")
        sub_question_id = request.form.get("sub_question_id")

        # Get all the captured images
        captured_images = {}
        for key in request.form:
            if key.startswith("captured_image_"):
                captured_images[key] = request.form[key]

        # Get the sub-question to check correct answer
        sub_question_ref = db.collection("sub_questions").document(sub_question_id).get()
        if sub_question_ref.exists:
            sub_question_data = sub_question_ref.to_dict()
            correct_answer = sub_question_data.get("correct_answer")
            points = sub_question_data.get("points", 1)

            # Initialize variables
            is_correct = False
            detected_value = None
            annotated_image_path = None
            
            # Process answer based on answer method
            if answer_method == AnswerMethod.ABACUS and captured_images:
                # Get the first captured image (we'll only use one for now)
                base64_image = next(iter(captured_images.values()))
                
                # Check answer using the abacus service
                is_correct, detected_value, annotated_image_path = check_abacus_answer(
                    base64_image, correct_answer
                )
            elif (answer_method == AnswerMethod.ANALOG_CLOCK or 
                  answer_method == AnswerMethod.DIGITAL_CLOCK) and captured_images:
                # Get the first captured image
                base64_image = next(iter(captured_images.values()))
                
                # Check answer using the clock service
                is_correct, detected_value, annotated_image_path = check_clock_answer(
                    base64_image, correct_answer
                )
            
            # If there's an annotated image path, copy it to the static folder
            if annotated_image_path and os.path.exists(annotated_image_path):
                static_uploads = os.path.join(current_app.static_folder, "uploads")
                os.makedirs(static_uploads, exist_ok=True)
                
                filename = os.path.basename(annotated_image_path)
                static_path = os.path.join(static_uploads, filename)
                shutil.copy(annotated_image_path, static_path)
                
                # Add the path to captured_images
                captured_images["annotated_image"] = f"/static/uploads/{filename}"
            
            # Save attempt with detection results
            result_data = {
                "detected_value": detected_value,
                "expected_value": correct_answer,
            }
            
            attempted = AttemptedQuestion(
                user_id=current_user.id,
                question_id=question_id,
                sub_question_id=sub_question_id,
                is_correct=is_correct,
                images=captured_images,
                result_data=result_data,  # Add results to the attempt record
            )
            attempted.save()

            # Update score if correct
            if is_correct:
                kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
                if kinesthetic_profile:
                    kinesthetic_profile.total_score += points
                    kinesthetic_profile.save()
                    
            # Pass the attempt ID to the result page
            return redirect(url_for('kinesthetic.submission_result', 
                                   attempted_question_id=attempted.id))

        # Update attempts counter
        kinesthetic_profile.current_lesson_attempts += 1

        # Check if lesson is complete (5 questions answered)
        if kinesthetic_profile.current_lesson_attempts >= 5:
            kinesthetic_profile.completed_lessons.append(subject)
            kinesthetic_profile.current_lesson_attempts = 0
            kinesthetic_profile.save()
            if len(kinesthetic_profile.completed_lessons) == len(Subject.CHOICES):
                return redirect(url_for("kinesthetic.all_lessons_completed"))
            return redirect(url_for("kinesthetic.user_home"))

        kinesthetic_profile.save()
        return redirect(url_for("kinesthetic.play", subject=subject))

    # Handle GET request
    # Get all available questions for all subjects
    questions_ref = (
        db.collection("questions")
        .where("is_published", "==", True)
        .get()
    )
    all_available_questions = [Question.from_doc(q) for q in questions_ref]

    # Check if there are any questions available
    if not all_available_questions:
        flash("No questions available.", "warning")
        return redirect(url_for("kinesthetic.user_home"))

    # Add default value for remaining_questions (out of 15 total)
    total_questions = 15
    remaining_questions = total_questions - (kinesthetic_profile.current_lesson_attempts or 0)

    if kinesthetic_profile.current_lesson_attempts >= total_questions:
        kinesthetic_profile.mixed_quiz_completed = True
        kinesthetic_profile.save()
        return redirect(url_for("kinesthetic.leaderboard"))

    # Group questions by subject to ensure we pick a good mix
    questions_by_subject = {}
    for question in all_available_questions:
        if question.subject not in questions_by_subject:
            questions_by_subject[question.subject] = []
        questions_by_subject[question.subject].append(question)

    # Select a random question, trying to balance subjects
    available_subjects = list(questions_by_subject.keys())
    
    # If we've already seen questions from this profile, try to pick 
    # from subjects we've seen less often
    subject_counts = kinesthetic_profile.subject_counts if hasattr(kinesthetic_profile, 'subject_counts') else {}
    
    # Default to a random subject if we can't determine which one to pick
    if not available_subjects:
        flash("No subjects have available questions.", "warning")
        return redirect(url_for("kinesthetic.user_home"))

    # Try to pick the subject with the lowest count
    min_count = float('inf')
    selected_subject = random.choice(available_subjects)
    
    for subject in available_subjects:
        count = subject_counts.get(subject, 0)
        if count < min_count and questions_by_subject[subject]:
            min_count = count
            selected_subject = subject
    
    # Now get a random question from the selected subject
    question = random.choice(questions_by_subject[selected_subject])
    
    # Update the subject counts for this profile
    if not hasattr(kinesthetic_profile, 'subject_counts'):
        kinesthetic_profile.subject_counts = {}
    
    kinesthetic_profile.subject_counts[selected_subject] = kinesthetic_profile.subject_counts.get(selected_subject, 0) + 1
    kinesthetic_profile.save()

    return render_template(
        "kinesthetic/play.html",
        question=question,
        subject=selected_subject,  # Pass the subject for the current question
        remaining_questions=remaining_questions,
    )


@kinesthetic_blueprint.route("/submission-result/<attempted_question_id>")
@login_required
def submission_result(attempted_question_id):
    # Get the attempted question from Firestore
    attempted_doc = db.collection("attempted_questions").document(attempted_question_id).get()
    
    if not attempted_doc.exists:
        flash("Attempt not found", "error")
        return redirect(url_for('kinesthetic.play'))
    
    attempted_data = attempted_doc.to_dict()
    
    # Get the question details
    question_doc = db.collection("questions").document(attempted_data.get("question_id", "")).get()
    sub_question_doc = db.collection("sub_questions").document(attempted_data.get("sub_question_id", "")).get()
    
    if not question_doc.exists or not sub_question_doc.exists:
        flash("Question details not found", "error")
        return redirect(url_for('kinesthetic.play'))
    
    question_data = question_doc.to_dict()
    sub_question_data = sub_question_doc.to_dict()
    
    # Prepare data for template
    attempted_question = {
        "id": attempted_question_id,
        "is_correct": attempted_data.get("is_correct", False),
        "images": attempted_data.get("images", {}),
        "result_data": attempted_data.get("result_data", {}),
        "question": {
            "id": question_doc.id,
            "text": question_data.get("text", ""),
            "html": question_data.get("text", ""),  # For compatibility with template
        },
        "sub_question": {
            "id": sub_question_doc.id,
            "text": sub_question_data.get("text", ""),
            "correct_answer": sub_question_data.get("correct_answer", ""),
        },
        "selected_choice": {  # Create a compatible structure for the template
            "html": f"Detected value: {attempted_data.get('result_data', {}).get('detected_value', 'Unknown')}",
            "is_correct": attempted_data.get("is_correct", False),
        }
    }
    
    return render_template(
        "kinesthetic/submission_result.html", attempted_question=attempted_question
    )


@kinesthetic_blueprint.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("kinesthetic.home"))

    form = UserLoginForm()
    if form.validate_on_submit():
        user = User.get_by_username(form.username.data)
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            return redirect(url_for("kinesthetic.user_home"))
        else:
            flash("Invalid username/password!", "danger")
    return render_template("kinesthetic/login.html", form=form, title="Login")


@kinesthetic_blueprint.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("kinesthetic.home"))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User.get_by_username(form.username.data)
        if user:
            flash("Username already exists")
            return redirect(url_for("kinesthetic.register"))

        password_hash = generate_password_hash(form.password.data)
        new_user = User(
            username=form.username.data,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            password_hash=password_hash,
        )

        new_user.save()

        kinesthetic_profile = QuizProfile(user_id=new_user.id)
        kinesthetic_profile.save()

        flash("Registration successful!")
        return redirect(url_for("kinesthetic.login"))

    return render_template("kinesthetic/registration.html", form=form, title="Register")


@kinesthetic_blueprint.route("/logout")
@login_required
def logout():
    """Redirect to main system logout."""
    # Clear the flask-login session
    logout_user()
   
    # Redirect to main system logout
    return redirect("http://localhost:5000/logout")


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


@kinesthetic_blueprint.route("/manage/questions")
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
        "kinesthetic/manage/questions_list.html",
        questions_by_subject=questions_by_subject,
        subjects=Subject.CHOICES,
    )


@kinesthetic_blueprint.route("/manage/questions/new", methods=["GET", "POST"])
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
        return redirect(url_for("kinesthetic.manage_questions"))
    return render_template(
        "kinesthetic/manage/question_form.html",
        form=form,
        title="New Question",
        initial_subject=subject,
    )


@kinesthetic_blueprint.route("/manage/questions/<question_id>", methods=["GET", "POST"])
@login_required  # Keep this decorator here
def edit_question(question_id):
    question_ref = db.collection("questions").document(question_id).get()
    if not question_ref.exists:
        flash("Question not found!", "error")
        return redirect(url_for("kinesthetic.manage_questions"))

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
            return redirect(url_for("kinesthetic.manage_questions"))
        except Exception as e:
            flash(f"Error updating question: {str(e)}", "error")
            return redirect(
                url_for("kinesthetic.edit_question", question_id=question_id)
            )

    return render_template(
        "kinesthetic/manage/question_form.html",
        form=form,
        question=question,
        title="Edit Question",
    )


@kinesthetic_blueprint.route(
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
        return redirect(url_for("kinesthetic.manage_questions"))  # Changed this line
    return render_template(
        "kinesthetic/manage/subquestion_form.html",
        form=form,
        question_id=question_id,
        title="New Sub-question",
    )


@kinesthetic_blueprint.route(
    "/manage/subquestions/<subquestion_id>", methods=["GET", "POST"]
)
def edit_subquestion(subquestion_id):
    subquestion_ref = db.collection("sub_questions").document(subquestion_id).get()
    if not subquestion_ref.exists:
        flash("Sub-question not found!", "error")
        return redirect(url_for("kinesthetic.manage_questions"))

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
        return redirect(url_for("kinesthetic.manage_questions"))  # Changed this line

    return render_template(
        "kinesthetic/manage/subquestion_form.html",
        form=form,
        subquestion=subquestion,
        title="Edit Sub-question",
    )


@kinesthetic_blueprint.route("/api/answer-methods/<subject>")
def get_answer_methods(subject):  # Remove @login_required decorator
    # Get the answer methods for the selected subject
    methods = Subject.ANSWER_METHODS.get(subject, [])
    return jsonify({"methods": methods})


@kinesthetic_blueprint.route("/manage/questions/<question_id>/delete", methods=["POST"])
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

    return redirect(url_for("kinesthetic.manage_questions"))


@kinesthetic_blueprint.route(
    "/manage/subquestions/<subquestion_id>/delete", methods=["POST"]
)
@login_required
def delete_subquestion(subquestion_id):
    try:
        db.collection("sub_questions").document(subquestion_id).delete()
        flash("Sub-question deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting sub-question: {str(e)}", "error")

    return redirect(url_for("kinesthetic.manage_questions"))


@kinesthetic_blueprint.route("/lesson-instructions/<subject>")
@login_required
def lesson_instructions(subject):
    subject_names = {
        "addition": "දශම පාඩම",
        "subtraction": "අඩු කිරීම පාඩම",
        "time": "කාලය පාඩම",
    }

    if subject not in subject_names:
        flash("Invalid subject selected", "error")
        return redirect(url_for("kinesthetic.user_home"))

    return render_template(
        "kinesthetic/lesson_instructions.html",
        subject=subject,
        subject_name=subject_names[subject],
    )


@kinesthetic_blueprint.route("/process-answer", methods=["POST"])
@login_required
def process_answer():
    """Process answer submission via AJAX"""
    question_id = request.form.get("question_pk")
    answer_method = request.form.get("answer_method")
    sub_question_id = request.form.get("sub_question_id")
    
    # Get all the captured images
    captured_images = {}
    for key in request.form:
        if key.startswith("captured_image_"):
            captured_images[key] = request.form[key]

    # Initialize response data
    response_data = {
        'is_correct': False,
        'detected_value': None,
        'expected_value': None,
        'annotated_image_url': None
    }
    
    # Get the sub-question to check correct answer
    sub_question_ref = db.collection("sub_questions").document(sub_question_id).get()
    if not sub_question_ref.exists:
        return jsonify(response_data), 400
    
    sub_question_data = sub_question_ref.to_dict()
    correct_answer = sub_question_data.get("correct_answer")
    points = sub_question_data.get("points", 1)
    
    # Process answer based on answer method
    if answer_method == AnswerMethod.ABACUS and captured_images:
        # Get the first captured image (we'll only use one for now)
        base64_image = next(iter(captured_images.values()))
        
        # Check answer using the abacus service
        is_correct, detected_value, annotated_image_path = check_abacus_answer(
            base64_image, correct_answer
        )
    elif (answer_method == AnswerMethod.ANALOG_CLOCK or 
          answer_method == AnswerMethod.DIGITAL_CLOCK) and captured_images:
        # Get the first captured image
        base64_image = next(iter(captured_images.values()))
        
        # Check answer using the clock service
        is_correct, detected_value, annotated_image_path = check_clock_answer(
            base64_image, correct_answer
        )
        
    # Update response data with detection results
    response_data['is_correct'] = is_correct
    response_data['detected_value'] = detected_value
    response_data['expected_value'] = correct_answer
    
    # If there's an annotated image path, copy it to the static folder
    if annotated_image_path and os.path.exists(annotated_image_path):
        static_uploads = os.path.join(current_app.static_folder, "uploads")
        os.makedirs(static_uploads, exist_ok=True)
        
        filename = os.path.basename(annotated_image_path)
        static_path = os.path.join(static_uploads, filename)
        shutil.copy(annotated_image_path, static_path)
        
        # Add the public URL to the response
        response_data['annotated_image_url'] = f"/static/uploads/{filename}"
        
        # Add the path to captured_images for database
        captured_images["annotated_image"] = f"/static/uploads/{filename}"
    
    # Save attempt with detection results
    result_data = {
        "detected_value": response_data['detected_value'],
        "expected_value": response_data['expected_value'],
    }
    
    attempted = AttemptedQuestion(
        user_id=current_user.id,
        question_id=question_id,
        sub_question_id=sub_question_id,
        is_correct=response_data['is_correct'],
        images=captured_images,
        result_data=result_data
    )
    attempted.save()

    # Update score if correct
    if response_data['is_correct']:
        kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
        if kinesthetic_profile:
            kinesthetic_profile.total_score += points
            kinesthetic_profile.save()
    
    # Update attempts counter
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    kinesthetic_profile.current_lesson_attempts += 1
    
    # Check if lesson is complete (5 questions answered)
    subject = request.args.get("subject", Subject.ADDITION)
    if kinesthetic_profile.current_lesson_attempts >= 5:
        kinesthetic_profile.completed_lessons.append(subject)
        kinesthetic_profile.current_lesson_attempts = 0
        response_data['lesson_completed'] = True
    
    kinesthetic_profile.save()
    return jsonify(response_data)


# @kinesthetic_blueprint.route("/process-all-answers", methods=["POST"])
# @login_required
# def process_all_answers():
#     """Process all answers for a question's sub-questions together"""
#     # Check for proper request
#     if request.method != "POST":
#         flash("Invalid request method", "error")
#         return redirect(url_for("kinesthetic.play"))
    
#     question_id = request.form.get("question_pk")
#     answer_method = request.form.get("answer_method")
#     sub_question_ids = request.form.getlist("sub_question_ids")
#     # Get subject from the form data
#     subject = request.form.get("subject", Subject.ADDITION)
    
#     # Initialize response
#     response_data = {
#         "success": True,
#         "results": [],
#         "subject": subject,
#         "redirect_url": url_for("kinesthetic.play")  # No subject needed for mixed quiz
#     }
    
#     total_points = 0
#     correct_count = 0
    
#     # Process each sub-question
#     for sub_question_id in sub_question_ids:
#         # Look for captured image for this sub-question
#         image_key = f"captured_image_{sub_question_id}"
#         base64_image = request.form.get(image_key)
        
#         if not base64_image:
#             continue  # Skip if no image for this sub-question
            
#         # Get sub-question details
#         sub_question_ref = db.collection("sub_questions").document(sub_question_id).get()
#         if not sub_question_ref.exists:
#             continue
            
#         sub_question_data = sub_question_ref.to_dict()
#         correct_answer = sub_question_data.get("correct_answer")
#         sub_question_text = sub_question_data.get("text", "")
#         points = sub_question_data.get("points", 1)
        
#         # Initialize result for this sub-question
#         result = {
#             "sub_question_id": sub_question_id,
#             "sub_question_text": sub_question_text,
#             "is_correct": False,
#             "detected_value": None,
#             "expected_value": correct_answer,
#             "annotated_image_url": None
#         }
        
#         # Process answer based on answer method
#         if answer_method == AnswerMethod.ABACUS:
#             # Check answer using the abacus service
#             is_correct, detected_value, annotated_image_path = check_abacus_answer(
#                 base64_image, correct_answer
#             )
#         elif answer_method == AnswerMethod.ANALOG_CLOCK or answer_method == AnswerMethod.DIGITAL_CLOCK:
#             # Check answer using the clock service
#             is_correct, detected_value, annotated_image_path = check_clock_answer(
#                 base64_image, correct_answer
#             )
#         else:
#             # Unsupported answer method
#             continue
            
#         # Update result data
#         result["is_correct"] = is_correct
#         result["detected_value"] = detected_value
        
#         # If there's an annotated image path, copy it to the static folder
#         if annotated_image_path and os.path.exists(annotated_image_path):
#             static_uploads = os.path.join(current_app.static_folder, "uploads")
#             os.makedirs(static_uploads, exist_ok=True)
            
#             filename = os.path.basename(annotated_image_path)
#             static_path = os.path.join(static_uploads, filename)
#             shutil.copy(annotated_image_path, static_path)
            
#             # Add the public URL to the result
#             result["annotated_image_url"] = f"/static/uploads/{filename}"
        
#         # Save attempt to database
#         captured_images = {image_key: base64_image}
#         if result["annotated_image_url"]:
#             captured_images["annotated_image"] = result["annotated_image_url"]
            
#         result_data = {
#             "detected_value": detected_value,
#             "expected_value": correct_answer,
#         }
        
#         attempted = AttemptedQuestion(
#             user_id=current_user.id,
#             question_id=question_id,
#             sub_question_id=sub_question_id,
#             is_correct=is_correct,
#             images=captured_images,
#             result_data=result_data
#         )
#         attempted.save()
        
#         # Update counters
#         if is_correct:
#             correct_count += 1
#             total_points += points
            
#         # Add to results list
#         response_data["results"].append(result)
    
#     # Update quiz profile
#     kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
#     if kinesthetic_profile:
#         # Add points to the profile
#         kinesthetic_profile.total_score += total_points
        
#         # Update attempts counter (this counts as one attempt regardless of sub-questions)
#         kinesthetic_profile.current_lesson_attempts += 1
        
#         # Check if all questions are complete (15 total)
#         total_questions = 15
#         if kinesthetic_profile.current_lesson_attempts >= total_questions:
#             kinesthetic_profile.mixed_quiz_completed = True
#             response_data["quiz_completed"] = True
#             response_data["redirect_url"] = url_for("kinesthetic.leaderboard")
            
#         kinesthetic_profile.save()
    
#     return jsonify(response_data)

def clean_base64_string(base64_string):
    """Clean and validate a base64 string, removing or replacing invalid characters"""
    if not base64_string:
        return None
        
    # Remove the data URL prefix if present
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    # Define valid base64 characters
    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    
    # Check for and log invalid characters
    invalid_chars = set()
    for i, c in enumerate(base64_string[:100]):
        if c not in valid_chars:
            invalid_chars.add(c)
    
    if invalid_chars:
        print(f"Found {len(invalid_chars)} invalid base64 characters: {list(invalid_chars)}")
        
        # Remove invalid characters
        base64_string = ''.join(c for c in base64_string if c in valid_chars)
        print(f"Cleaned base64 string, new length: {len(base64_string)}")
    
    # Add padding if needed
    padding_needed = len(base64_string) % 4
    if padding_needed > 0:
        base64_string += '=' * (4 - padding_needed)
        
    return base64_string

@kinesthetic_blueprint.route("/process-all-answers", methods=["POST"])
@login_required
def process_all_answers():
    """Process all answers for a question's sub-questions together"""
    # Check for proper request
    if request.method != "POST":
        flash("Invalid request method", "error")
        return redirect(url_for("kinesthetic.play"))
    
    question_id = request.form.get("question_pk")
    answer_method = request.form.get("answer_method")
    sub_question_ids = request.form.getlist("sub_question_ids")
    # Get subject from the form data
    subject = request.form.get("subject", Subject.ADDITION)
    
    # Initialize response
    response_data = {
        "success": True,
        "results": [],
        "subject": subject,
        "redirect_url": url_for("kinesthetic.play")  # No subject needed for mixed quiz
    }
    
    total_points = 0
    correct_count = 0
    
    # Process each sub-question
    for sub_question_id in sub_question_ids:
        try:
            # Look for captured image for this sub-question
            image_key = f"captured_image_{sub_question_id}"
            base64_image = request.form.get(image_key)
            
            if not base64_image:
                print(f"No image data for sub-question {sub_question_id}")
                continue  # Skip if no image for this sub-question
            
            # Basic validation of base64 data length
            if len(base64_image) < 100:
                print(f"Image data too small for sub-question {sub_question_id}: {len(base64_image)} bytes")
                continue
                
            # Store the original image for saving to database later
            original_image = base64_image
            
            # Get sub-question details
            sub_question_ref = db.collection("sub_questions").document(sub_question_id).get()
            if not sub_question_ref.exists:
                continue
                
            sub_question_data = sub_question_ref.to_dict()
            correct_answer = sub_question_data.get("correct_answer")
            sub_question_text = sub_question_data.get("text", "")
            points = sub_question_data.get("points", 1)
            
            # Initialize result for this sub-question
            result = {
                "sub_question_id": sub_question_id,
                "sub_question_text": sub_question_text,
                "is_correct": False,
                "detected_value": None,
                "expected_value": correct_answer,
                "annotated_image_url": None
            }
            
            try:
                # Process answer based on answer method
                if answer_method == AnswerMethod.ABACUS:
                    # Check answer using the abacus service
                    is_correct, detected_value, annotated_image_path = check_abacus_answer(
                        base64_image, correct_answer
                    )
                elif answer_method == AnswerMethod.ANALOG_CLOCK or answer_method == AnswerMethod.DIGITAL_CLOCK:
                    # Clean and validate the base64 string
                    processed_base64 = clean_base64_string(base64_image)
                    
                    if not processed_base64:
                        print(f"Failed to clean base64 data for sub-question {sub_question_id}")
                        result["detected_value"] = "Error: Invalid image data"
                        response_data["results"].append(result)
                        continue
                    
                    # Check answer using the clock service with cleaned base64 data
                    is_correct, detected_value, annotated_image_path = check_clock_answer(
                        processed_base64, correct_answer
                    )
                else:
                    # Unsupported answer method
                    continue
                    
            except Exception as e:
                print(f"Error processing image for sub-question {sub_question_id}: {str(e)}")
                result["detected_value"] = f"Error: {str(e)}"
                response_data["results"].append(result)
                continue
                
            # Update result data
            result["is_correct"] = is_correct
            result["detected_value"] = detected_value
            
            # If there's an annotated image path, copy it to the static folder
            if annotated_image_path and os.path.exists(annotated_image_path):
                static_uploads = os.path.join(current_app.static_folder, "uploads")
                os.makedirs(static_uploads, exist_ok=True)
                
                filename = os.path.basename(annotated_image_path)
                static_path = os.path.join(static_uploads, filename)
                shutil.copy(annotated_image_path, static_path)
                
                # Add the public URL to the result
                result["annotated_image_url"] = f"/static/uploads/{filename}"
            
            # Save attempt to database - use the original base64 image with prefix intact
            captured_images = {image_key: original_image}
            if result.get("annotated_image_url"):
                captured_images["annotated_image"] = result["annotated_image_url"]
                
            result_data = {
                "detected_value": detected_value,
                "expected_value": correct_answer,
            }
            
            attempted = AttemptedQuestion(
                user_id=current_user.id,
                question_id=question_id,
                sub_question_id=sub_question_id,
                is_correct=is_correct,
                images=captured_images,
                result_data=result_data
            )
            attempted.save()
            
            # Update counters
            if is_correct:
                correct_count += 1
                total_points += points
                
            # Add to results list
            response_data["results"].append(result)
        
        except Exception as e:
            print(f"Error processing sub-question {sub_question_id}: {str(e)}")
            # Continue with other sub-questions even if one fails
            continue
    
    # Update quiz profile
    try:
        kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
        if kinesthetic_profile:
            # Add points to the profile
            kinesthetic_profile.total_score += total_points
            
            # Update attempts counter (this counts as one attempt regardless of sub-questions)
            kinesthetic_profile.current_lesson_attempts += 1
            
            # Check if all questions are complete (15 total)
            total_questions = 15
            if kinesthetic_profile.current_lesson_attempts >= total_questions:
                kinesthetic_profile.mixed_quiz_completed = True
                response_data["quiz_completed"] = True
                response_data["redirect_url"] = url_for("kinesthetic.leaderboard")
                
            kinesthetic_profile.save()
    except Exception as e:
        print(f"Error updating profile: {str(e)}")
        # Continue anyway, so we at least return results to the user
    
    return jsonify(response_data)

@kinesthetic_blueprint.route("/save-captured-image", methods=["POST"])
@login_required
def save_captured_image():
    """Save a captured image to the static/uploads directory"""
    if not request.is_json:
        return jsonify({"success": False, "error": "Invalid request format"}), 400
    
    data = request.get_json()
    image_data = data.get("image_data")
    filename = data.get("filename")
    
    if not image_data or not filename:
        return jsonify({"success": False, "error": "Missing image data or filename"}), 400
    
    # Make sure we have valid base64 data
    if "base64," in image_data:
        # Split the base64 string to get only the data part
        image_data = image_data.split("base64,")[1]
    
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(current_app.static_folder, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate a unique filename with user ID to avoid conflicts
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{current_user.id}_{timestamp}_{filename}"
        
        # Create the full file path
        file_path = os.path.join(uploads_dir, unique_filename)
        
        # Save the image file
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        
        # Return the URL path that can be used to access the file
        file_url = f"/static/uploads/{unique_filename}"
        
        return jsonify({
            "success": True, 
            "file_path": file_url,
            "message": "Image saved successfully"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@kinesthetic_blueprint.route("/save-and-process-clock", methods=["POST"])
@login_required
def save_and_process_clock():
    """Save a clock drawing to the static/uploads directory and process it with the clock model"""
    if not request.is_json:
        return jsonify({"success": False, "error": "Invalid request format"}), 400
    
    data = request.get_json()
    image_data = data.get("image_data")
    filename = data.get("filename")
    sub_question_id = data.get("sub_question_id")
    
    if not image_data or not filename:
        return jsonify({"success": False, "error": "Missing image data or filename"}), 400
    
    try:
        # Make sure we have valid base64 data
        if "base64," in image_data:
            # Split the base64 string to get only the data part
            image_data = image_data.split("base64,")[1]
        
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(current_app.static_folder, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate a unique filename with user ID to avoid conflicts
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{current_user.id}_{timestamp}_{filename}"
        
        # Create the full file path
        file_path = os.path.join(uploads_dir, unique_filename)
        
        # Save the image file
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        
        # Process with clock detection model
        detected_result = None
        expected_answer = None
        is_correct = False
        annotated_image_url = None
        
        # Get the sub-question to find expected answer
        if sub_question_id:
            sub_question_ref = db.collection("sub_questions").document(sub_question_id).get()
            if sub_question_ref.exists:
                sub_question_data = sub_question_ref.to_dict()
                expected_answer = sub_question_data.get("correct_answer", "")
                
                # Process with clock model
                from services.clock_service import check_clock_answer
                
                # Convert file path to base64 for the clock service
                with open(file_path, "rb") as image_file:
                    base64_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Process with clock model
                is_correct, detected_time, annotated_path = check_clock_answer(
                    base64_data, expected_answer
                )
                
                # If we have an annotated image, save its URL
                if annotated_path and os.path.exists(annotated_path):
                    annotated_filename = f"annotated_{unique_filename}"
                    annotated_dest = os.path.join(uploads_dir, annotated_filename)
                    shutil.copy(annotated_path, annotated_dest)
                    annotated_image_url = f"/static/uploads/{annotated_filename}"
                
                detected_result = detected_time
        
        # Return the public URL path that can be used to access the file
        file_url = f"/static/uploads/{unique_filename}"
        
        return jsonify({
            "success": True, 
            "file_path": file_url,
            "detected_time": detected_result,
            "expected_time": expected_answer,
            "is_correct": is_correct,
            "annotated_image_url": annotated_image_url,
            "message": "Image saved and processed successfully"
        })
        
    except Exception as e:
        print(f"Error saving/processing clock image: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

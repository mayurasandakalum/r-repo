from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify, current_app, session
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from firebase_admin import firestore, auth
from datetime import datetime
import random
import base64
import os
import shutil
import requests
import json

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

# New route to handle authentication from main system
@kinesthetic_blueprint.route("/authenticate")
def authenticate():
    token = request.args.get('token')
    if not token:
        flash('Authentication token missing', 'error')
        return redirect('http://localhost:5000/login')
    
    try:
        # For custom tokens, we need to extract the user ID directly
        # Custom tokens are in the format: header.payload.signature
        import base64
        import json
        
        # Split the token to get the payload part (second part)
        parts = token.split('.')
        if len(parts) != 3:
            flash('Invalid token format', 'error')
            return redirect('http://localhost:5000/login')
            
        # Decode the payload
        payload = parts[1]
        # Add padding if needed
        padding = '=' * (4 - len(payload) % 4) if len(payload) % 4 != 0 else ''
        payload_padded = payload + padding
        
        try:
            decoded = base64.urlsafe_b64decode(payload_padded)
            payload_data = json.loads(decoded)
            
            # Extract the user ID from the custom token payload
            # The structure is different from ID tokens
            user_id = payload_data.get('uid')
            
            if not user_id:
                flash('Could not extract user ID from token', 'error')
                return redirect('http://localhost:5000/login')
                
        except Exception as e:
            print(f"Error decoding token: {str(e)}")
            flash('Invalid token format', 'error')
            return redirect('http://localhost:5000/login')
        
        # Rest of the authenticate function remains the same
        # Check if user exists in kinesthetic database
        user = User.get_by_id(user_id)
        if not user:
            # Fetch user details from main system
            response = requests.get(f'http://localhost:5000/api/user/{user_id}')
            if response.status_code != 200:
                flash('Failed to fetch user details', 'error')
                return redirect('http://localhost:5000/login')
            
            user_data = response.json()
            # Create user in kinesthetic database
            username = user_data['email'].split('@')[0] if '@' in user_data['email'] else user_data['email']
            user = User(
                username=username,
                email=user_data['email'],
                first_name=user_data.get('name', '').split()[0] if user_data.get('name') else '',
                last_name=user_data.get('name', '').split()[-1] if user_data.get('name') and len(user_data.get('name').split()) > 1 else '',
                password_hash='',  # No password needed since login is removed
                id=user_id  # Use the same ID as main system
            )
            user.save()
        
        # Log in the user with Flask-Login
        login_user(user)
        
        # Create QuizProfile if it doesn't exist
        kinesthetic_profile = QuizProfile.get_by_user_id(user_id)
        if not kinesthetic_profile:
            kinesthetic_profile = QuizProfile(user_id=user_id)
            kinesthetic_profile.save()
        
        return redirect(url_for('kinesthetic.user_home'))
    
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        flash(f'Authentication failed: {str(e)}', 'error')
        return redirect('http://localhost:5000/login')


@kinesthetic_blueprint.route("/")
def home():
    return render_template("kinesthetic/home.html")


@kinesthetic_blueprint.route("/user-home")
@login_required
def user_home():
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    
    # Get attempts by quiz type for performance comparison
    attempts_by_type = {}
    weakest_subject = None
    
    if kinesthetic_profile:
        # First check if we already have a stored weakest subject
        weakest_subject = kinesthetic_profile.weakest_subject
        
        # If no stored value exists or we need to recalculate
        if not weakest_subject and kinesthetic_profile.subject_performance:
            # Only calculate if we have performance data
            if len(kinesthetic_profile.subject_performance) > 0:
                weakest_data = kinesthetic_profile.get_weakest_subject()
                weakest_subject = weakest_data.get("subject")
                
                # Store the weakest subject in the profile for future use
                if weakest_subject and not kinesthetic_profile.weakest_subject:
                    kinesthetic_profile.weakest_subject = weakest_subject
                    kinesthetic_profile.save()
        
        if weakest_subject:
            # Get attempts for this subject and split by quiz_type
            attempts_query = (
                db.collection("attempted_questions")
                .where("user_id", "==", current_user.id)
                .get()
            )
            
            # Process the attempts to get statistics by quiz_type
            for attempt in attempts_query:
                attempt_data = attempt.to_dict()
                quiz_type = attempt_data.get("quiz_type", "mixed_quiz")  # Default if not specified
                
                # Get the question to determine subject
                question_id = attempt_data.get("question_id")
                if not question_id:
                    continue
                    
                question_ref = db.collection("questions").document(question_id).get()
                if not question_ref.exists:
                    continue
                    
                question_data = question_ref.to_dict()
                subject = question_data.get("subject")
                
                if subject != weakest_subject:
                    continue
                
                # Initialize counters if needed
                if quiz_type not in attempts_by_type:
                    attempts_by_type[quiz_type] = {"total": 0, "correct": 0}
                
                # Count the attempt
                attempts_by_type[quiz_type]["total"] += 1
                if attempt_data.get("is_correct", False):
                    attempts_by_type[quiz_type]["correct"] += 1
    
    return render_template(
        "kinesthetic/user_home.html",
        kinesthetic_profile=kinesthetic_profile,
        attempts_by_type=attempts_by_type,
        weakest_subject=weakest_subject  # Pass the weakest subject directly to the template
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
                quiz_type="mixed_quiz"  # Set quiz_type to "mixed_quiz"
            )
            attempted.save()

            # Update score if correct
            if is_correct:
                kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
                if kinesthetic_profile:
                    kinesthetic_profile.total_score += points
                    kinesthetic_profile.save()
                    
                    # # Sync marks with main system
                    # try:
                    #     requests.post('http://localhost:5000/api/save_marks', json={
                    #         'user_id': current_user.id,
                    #         'quiz_id': question_id,
                    #         'score': points,
                    #         'subject': kinesthetic_profile.subject_counts
                    #     }, headers={'Content-Type': 'application/json'})
                    # except Exception as e:
                    #     flash(f'Failed to sync marks with main system: {str(e)}', 'warning')
                    
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
        "addition": "එකතු කිරීම පාඩම",
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
    
    # Get the question to determine its subject
    question_ref = db.collection("questions").document(question_id).get()
    if not question_ref.exists:
        return jsonify({"success": False, "error": "Question not found"}), 400
    
    question_data = question_ref.to_dict()
    question_subject = question_data.get("subject", Subject.ADDITION)
    
    # Process each sub-question
    for sub_question_id in sub_question_ids:
        # Look for captured image for this sub-question
        image_key = f"captured_image_{sub_question_id}"
        base64_image = request.form.get(image_key)
        
        if not base64_image:
            continue  # Skip if no image for this sub-question
            
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
        
        # Process answer based on answer method
        if answer_method == AnswerMethod.ABACUS:
            # Check answer using the abacus service
            is_correct, detected_value, annotated_image_path = check_abacus_answer(
                base64_image, correct_answer
            )
        elif answer_method == AnswerMethod.ANALOG_CLOCK or answer_method == AnswerMethod.DIGITAL_CLOCK:
            # Check answer using the clock service
            is_correct, detected_value, annotated_image_path = check_clock_answer(
                base64_image, correct_answer
            )
        else:
            # Unsupported answer method
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
        
        # Save attempt to database
        captured_images = {image_key: base64_image}
        if result["annotated_image_url"]:
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
    
    # Update quiz profile
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    if kinesthetic_profile:
        # Add points to the profile
        kinesthetic_profile.total_score += total_points
        
        # Update performance tracking by subject
        if question_subject not in kinesthetic_profile.subject_performance:
            kinesthetic_profile.subject_performance[question_subject] = {"correct": 0, "total": 0, "score": 0}
        
        # Update the subject performance - count each sub-question as an attempt
        kinesthetic_profile.subject_performance[question_subject]["total"] += len(response_data["results"])
        kinesthetic_profile.subject_performance[question_subject]["correct"] += correct_count
        
        # Add the score - check if 'score' key exists first (for backward compatibility)
        if "score" not in kinesthetic_profile.subject_performance[question_subject]:
            kinesthetic_profile.subject_performance[question_subject]["score"] = 0
        kinesthetic_profile.subject_performance[question_subject]["score"] += total_points
        
        # Sync marks with main system if points were earned
        if total_points > 0:
            try:
                requests.post('http://localhost:5000/api/save_marks', json={
                    'user_id': current_user.id,
                    'quiz_id': question_id,
                    'score': total_points,
                    'subject': kinesthetic_profile.subject_counts
                }, headers={'Content-Type': 'application/json'})
            except Exception as e:
                print(f'Failed to sync marks with main system: {str(e)}')
        
        # Update attempts counter (this counts as one attempt regardless of sub-questions)
        kinesthetic_profile.current_lesson_attempts += 1
        
        # Check if all questions are complete (15 total)
        total_questions = 15
        if kinesthetic_profile.current_lesson_attempts >= total_questions:
            kinesthetic_profile.mixed_quiz_completed = True
            
            # Store the weakest subject when the quiz is completed
            if not kinesthetic_profile.weakest_subject:
                weakest_data = kinesthetic_profile.get_weakest_subject()
                kinesthetic_profile.weakest_subject = weakest_data.get("subject")
            
            response_data["quiz_completed"] = True
            response_data["redirect_url"] = url_for("kinesthetic.user_home")  # Changed to user_home to show results
            
        kinesthetic_profile.save()
    
    return jsonify(response_data)


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

# Add this route to redirect to main system logout

@kinesthetic_blueprint.route("/logout")
def logout():
    """Redirect to main system logout."""
    # Clear the flask-login session
    logout_user()
    
    # Redirect to main system logout
    return redirect("http://localhost:5000/logout")


@kinesthetic_blueprint.route("/subject-help/<subject>")
@login_required
def subject_help(subject):
    """Show tutorial video for the specified subject."""
    # Validate subject
    if subject not in [Subject.ADDITION, Subject.SUBTRACTION, Subject.TIME]:
        flash("Invalid subject specified", "error")
        return redirect(url_for("kinesthetic.user_home"))
        
    # Get subject name for display
    subject_names = {
        "addition": "එකතු කිරීම",
        "subtraction": "අඩු කිරීම",
        "time": "කාලය",
    }
    
    # YouTube video IDs for each subject
    video_ids = {
        "addition": "Cdx4fyEe3RY",      
        "subtraction": "wUhllXSYzd0",   
        "time": "EUg4HF2JoC0",          
    }
    
    # Get user profile to check if video has already been watched
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    video_already_watched = False
    
    if kinesthetic_profile and hasattr(kinesthetic_profile, 'watched_videos'):
        video_already_watched = subject in kinesthetic_profile.watched_videos
    
    return render_template(
        "kinesthetic/subject_help.html",
        subject=subject,
        subject_name=subject_names.get(subject, subject),
        video_id=video_ids.get(subject),
        video_already_watched=video_already_watched
    )

@kinesthetic_blueprint.route("/weakest-subject-quiz")
@login_required
def weakest_subject_quiz():
    """Start a 5-question quiz for the user's weakest subject."""
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    if not kinesthetic_profile:
        flash("Quiz profile not found.", "error")
        return redirect(url_for("kinesthetic.user_home"))

    weakest_subject_data = kinesthetic_profile.get_weakest_subject()
    weakest_subject = weakest_subject_data.get("subject")

    if not weakest_subject:
        flash("Could not determine weakest subject.", "warning")
        return redirect(url_for("kinesthetic.user_home"))

    # Get questions for the weakest subject
    questions_ref = (
        db.collection("questions")
        .where("subject", "==", weakest_subject)
        .where("is_published", "==", True)
        .limit(5)
        .get()
    )
    
    available_questions = [Question.from_doc(q) for q in questions_ref]
    
    # Check if there are any questions available
    if not available_questions:
        flash(f"No questions available for {weakest_subject}.", "warning")
        return redirect(url_for("kinesthetic.user_home"))
    
    # Store initial score in session for comparison later
    session['initial_score'] = kinesthetic_profile.total_score
    session['weakest_subject'] = weakest_subject
    
    # Store question IDs in session to maintain the set of 5 questions
    session['weakest_subject_question_ids'] = [q.id for q in available_questions]
    session['weakest_subject_current_question'] = 0
    session['weakest_subject_remaining'] = 5
    
    # Get the first question
    question = available_questions[0]
    
    # Get sub-questions for the question
    sub_questions = SubQuestion.get_by_question(question.id)
    question._sub_questions = sub_questions

    return render_template(
        "kinesthetic/weakest_subject_quiz.html",
        question=question,
        subject=weakest_subject,
        remaining_questions=5
    )

@kinesthetic_blueprint.route("/next-weakest-subject-question")
@login_required
def next_weakest_subject_question():
    """Get the next question for the weakest subject quiz."""
    # Check if there are question IDs in the session
    question_ids = session.get('weakest_subject_question_ids', [])
    current_question_index = session.get('weakest_subject_current_question', 0)
    remaining = session.get('weakest_subject_remaining', 0)
    
    if not question_ids or current_question_index >= len(question_ids) - 1 or remaining <= 1:
        # Quiz is completed, show results
        return redirect(url_for('kinesthetic.process_weakest_subject_quiz'))
    
    # Move to the next question
    current_question_index += 1
    session['weakest_subject_current_question'] = current_question_index
    session['weakest_subject_remaining'] = remaining - 1
    
    # Get the next question
    question_id = question_ids[current_question_index]
    question_ref = db.collection("questions").document(question_id).get()
    
    if not question_ref.exists:
        flash("Question not found.", "error")
        return redirect(url_for("kinesthetic.user_home"))
    
    question = Question.from_doc(question_ref)
    
    # Get sub-questions
    sub_questions = SubQuestion.get_by_question(question.id)
    question._sub_questions = sub_questions
    
    return render_template(
        "kinesthetic/weakest_subject_quiz.html",
        question=question,
        subject=session.get('weakest_subject'),
        remaining_questions=remaining
    )

@kinesthetic_blueprint.route("/process-weakest-subject-quiz", methods=["POST", "GET"])
@login_required
def process_weakest_subject_quiz():
    """Process the quiz for the weakest subject and compare scores."""
    # If it's a GET request, show the final results
    if request.method == "GET":
        # Get initial score from session
        initial_score = session.pop('initial_score', 0)
        subject = session.pop('weakest_subject', Subject.ADDITION)
        
        # Clear other session variables
        session.pop('weakest_subject_question_ids', None)
        session.pop('weakest_subject_current_question', None)
        session.pop('weakest_subject_remaining', None)
        
        # Get current profile score
        kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
        final_score = kinesthetic_profile.total_score
        score_difference = final_score - initial_score
        
        # Get detailed statistics for both quiz types
        # For the mixed quiz (initial)
        initial_stats = {
            "correct": 0,
            "total": 0,
            "percentage": 0,
            "score": 0
        }
        
        # For the weakest subject quiz (final)
        final_stats = {
            "correct": 0,
            "total": 0,
            "percentage": 0,
            "score": 0
        }
        
        # Get mixed quiz stats from subject_performance
        if subject in kinesthetic_profile.subject_performance:
            perf_data = kinesthetic_profile.subject_performance[subject]
            initial_stats["correct"] = perf_data.get("correct", 0)
            initial_stats["total"] = perf_data.get("total", 0)
            initial_stats["score"] = perf_data.get("score", 0)
            if initial_stats["total"] > 0:
                initial_stats["percentage"] = (initial_stats["correct"] / initial_stats["total"]) * 100
                
        # Get weakest subject quiz stats from weakest_subject_performance
        if subject in kinesthetic_profile.weakest_subject_performance:
            perf_data = kinesthetic_profile.weakest_subject_performance[subject]
            final_stats["correct"] = perf_data.get("correct", 0)
            final_stats["total"] = perf_data.get("total", 0)
            final_stats["score"] = perf_data.get("score", 0)
            if final_stats["total"] > 0:
                final_stats["percentage"] = (final_stats["correct"] / final_stats["total"]) * 100
        
        # Store this comparison data in the database
        if not hasattr(kinesthetic_profile, 'quiz_comparisons'):
            kinesthetic_profile.quiz_comparisons = {}
        
        kinesthetic_profile.quiz_comparisons[subject] = {
            "before": initial_stats,
            "after": final_stats,
            "initial_score": initial_score,
            "final_score": final_score,
            "score_difference": score_difference,
            "timestamp": datetime.utcnow()
        }
        
        # Save the updated profile
        kinesthetic_profile.save()
        
        # Render the result page
        return render_template(
            "kinesthetic/weakest_subject_quiz_result.html",
            initial_score=initial_score,
            final_score=final_score,
            score_difference=score_difference,
            subject=subject,
            initial_stats=initial_stats,
            final_stats=final_stats
        )
    
    # For POST requests, process the current question's answers
    question_id = request.form.get("question_pk")
    answer_method = request.form.get("answer_method")
    sub_question_ids = request.form.getlist("sub_question_ids")
    
    # Get subject from session
    subject = session.get('weakest_subject', Subject.ADDITION)
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    # Initialize response
    response_data = {
        "success": True,
        "results": [],
        "subject": subject,
        "redirect_url": url_for("kinesthetic.next_weakest_subject_question")
    }
    
    total_points = 0
    correct_count = 0
    
    # Process each sub-question (similar to process_all_answers)
    for sub_question_id in sub_question_ids:
        # Look for captured image for this sub-question
        image_key = f"captured_image_{sub_question_id}"
        base64_image = request.form.get(image_key)
        
        if not base64_image:
            continue  # Skip if no image for this sub-question
            
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
        
        # Process answer based on answer method
        if answer_method == AnswerMethod.ABACUS:
            # Check answer using the abacus service
            is_correct, detected_value, annotated_image_path = check_abacus_answer(
                base64_image, correct_answer
            )
        elif answer_method == AnswerMethod.ANALOG_CLOCK or answer_method == AnswerMethod.DIGITAL_CLOCK:
            # Check answer using the clock service
            is_correct, detected_value, annotated_image_path = check_clock_answer(
                base64_image, correct_answer
            )
        else:
            # Unsupported answer method
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
        
        # Save attempt to database
        captured_images = {image_key: base64_image}
        if result["annotated_image_url"]:
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
            result_data=result_data,
            quiz_type="weakest_subject"  # Set quiz_type to "weakest_subject"
        )
        attempted.save()
        
        # Update counters
        if is_correct:
            correct_count += 1
            total_points += points
            
        # Add to results list
        response_data["results"].append(result)
    
    # Update quiz profile
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    if kinesthetic_profile:
        # Add points to the profile
        kinesthetic_profile.total_score += total_points
        
        # Initialize weakest_subject_performance if it doesn't exist
        if not hasattr(kinesthetic_profile, 'weakest_subject_performance'):
            kinesthetic_profile.weakest_subject_performance = {}
            
        # Update performance tracking by subject in weakest_subject_performance instead of subject_performance
        if subject not in kinesthetic_profile.weakest_subject_performance:
            kinesthetic_profile.weakest_subject_performance[subject] = {"correct": 0, "total": 0, "score": 0}
        
        # Update the weakest subject performance
        kinesthetic_profile.weakest_subject_performance[subject]["total"] += len(response_data["results"])
        kinesthetic_profile.weakest_subject_performance[subject]["correct"] += correct_count
        kinesthetic_profile.weakest_subject_performance[subject]["score"] += total_points
        
        kinesthetic_profile.save()
    
    # Update the remaining questions counter
    remaining = session.get('weakest_subject_remaining', 0) - 1
    session['weakest_subject_remaining'] = remaining

    # If this was the last question, redirect to the results page
    if remaining <= 0:
        response_data["redirect_url"] = url_for("kinesthetic.process_weakest_subject_quiz")

    # Return JSON response for AJAX requests
    if is_ajax:
        return jsonify(response_data)
    else:
        # Get the next question or finish the quiz
        return redirect(url_for('kinesthetic.next_weakest_subject_question'))

@kinesthetic_blueprint.route("/api/video-watched/<subject>", methods=["POST"])
@login_required
def mark_video_as_watched(subject):
    """Mark a video as watched for the current user."""
    # Validate subject
    if subject not in [Subject.ADDITION, Subject.SUBTRACTION, Subject.TIME]:
        return jsonify({"success": False, "message": "Invalid subject specified"})
        
    # Get user profile
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    if not kinesthetic_profile:
        kinesthetic_profile = QuizProfile(user_id=current_user.id)
    
    # Initialize watched_videos list if it doesn't exist
    if not hasattr(kinesthetic_profile, 'watched_videos') or kinesthetic_profile.watched_videos is None:
        kinesthetic_profile.watched_videos = []
    
    # Add subject to watched videos if not already there
    if subject not in kinesthetic_profile.watched_videos:
        kinesthetic_profile.watched_videos.append(subject)
        kinesthetic_profile.save()
        
    return jsonify({
        "success": True, 
        "message": "Video marked as watched", 
        "watched_videos": kinesthetic_profile.watched_videos
    })

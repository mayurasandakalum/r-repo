"""
Route handlers and view functions.
"""

from flask import render_template, jsonify, redirect, request, session, url_for, flash
import requests
import config
import psutil
import datetime
import firebase_admin
from firebase_admin import credentials, auth, firestore
from models import (
    login_required,
    teacher_required,
    student_required,
    User,
    Teacher,
    Student,
)
import os
import json
import sys
from flask import send_from_directory
from retrieve_students import retrieve_students_data
from classify_students import main as classify_main

# Initialize the kinesthetic Firebase app with its service account
kinesthetic_cred_path = "./kinesthetic/serviceAccountKey.json"
try:
    kinesthetic_cred = credentials.Certificate(kinesthetic_cred_path)
    kinesthetic_app = firebase_admin.initialize_app(
        kinesthetic_cred, name="kinesthetic"
    )
except (ValueError, firebase_admin.exceptions.FirebaseError) as e:
    print(f"Error initializing kinesthetic app: {str(e)}")
    # If it's already initialized, get the existing app
    try:
        kinesthetic_app = firebase_admin.get_app(name="kinesthetic")
    except:
        print("Could not get existing kinesthetic app")

# Initialize the audio Firebase app with its service account
audio_cred_path = "./audio/learn-pal-firebase-adminsdk-ugedp-fcb865a7d8.json"
try:
    audio_cred = credentials.Certificate(audio_cred_path)
    audio_app = firebase_admin.initialize_app(audio_cred, name="audio")
except (ValueError, firebase_admin.exceptions.FirebaseError) as e:
    print(f"Error initializing audio app: {str(e)}")
    # If it's already initialized, get the existing app
    try:
        audio_app = firebase_admin.get_app(name="audio")
    except:
        print("Could not get existing audio app")

# Initialize the readwrite Firebase app with its service account
readwrite_cred_path = "./readwrite/learn-pal-firebase-adminsdk-ugedp-fcb865a7d8.json"
try:
    readwrite_cred = credentials.Certificate(readwrite_cred_path)
    readwrite_app = firebase_admin.initialize_app(readwrite_cred, name="readwrite")
except (ValueError, firebase_admin.exceptions.FirebaseError) as e:
    print(f"Error initializing readwrite app: {str(e)}")
    # If it's already initialized, get the existing app
    try:
        readwrite_app = firebase_admin.get_app(name="readwrite")
    except:
        print("Could not get existing readwrite app")


def init_routes(app):
    """Initialize all routes for the Flask app."""

    @app.route("/")
    def index():
        """Main app index showing links to all sub-apps."""
        user_id = session.get("user_id")
        user_type = session.get("user_type")

        # If teacher is logged in, redirect to dashboard
        if user_id and user_type == "teacher":
            return redirect(url_for("dashboard"))
        elif user_id and user_type == "student":
            return redirect(url_for("student_dashboard"))

        return render_template(
            "home.html",
            kinesthetic_url=f"http://localhost:{config.KINESTHETIC_APP_PORT}",
            readwrite_url=f"http://localhost:{config.READWRITE_APP_PORT}",
            visual_url=f"http://localhost:{config.VISUAL_APP_PORT}",
            audio_url=f"http://localhost:{config.AUDIO_APP_PORT}",
            user_id=user_id,
            user_type=user_type,
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        """Handle user login."""
        # Debug line to check if we're processing a form submission
        print(
            f"Login request: {request.method}, form data: {request.form if request.method == 'POST' else 'N/A'}"
        )

        if request.method == "POST":
            email = request.form.get("email")
            password = request.form.get("password")
            user_type = request.form.get("user_type")

            if not email or not password or not user_type:
                flash("Please fill in all fields")
                return render_template("login.html")

            try:
                # Verify with Firebase Authentication
                user = User.get_by_email(email)

                # Verify user role matches the selected type
                if user_type == "teacher":
                    # Check if the user exists as a teacher
                    teacher = Teacher.get_by_email(email)
                    if not teacher:
                        flash("This account is not registered as a teacher")
                        return render_template("login.html")

                    # Store user info in session ONLY if user is a verified teacher
                    session["user_id"] = user.uid
                    session["email"] = user.email
                    session["user_type"] = "teacher"

                    # Check if we need to create basic teacher data
                    teacher = Teacher.get(user.uid)
                    if not teacher:
                        # First time login after registration - store basic info
                        Teacher.create_basic(user.uid, user.email)

                    return redirect(url_for("dashboard"))

                elif user_type == "student":
                    # Check if the user exists as a student
                    student = Student.get_by_email(email)
                    if not student:
                        flash("This account is not registered as a student")
                        return render_template("login.html")

                    # Store user info in session ONLY if user is a verified student
                    session["user_id"] = user.uid
                    session["email"] = user.email
                    session["user_type"] = "student"

                    # Generate and store kinesthetic token in session for later use
                    try:
                        kinesthetic_token = auth.create_custom_token(
                            user.uid, app=kinesthetic_app
                        )
                        kinesthetic_token_str = kinesthetic_token.decode(
                            "utf-8"
                        )  # Convert bytes to string
                        session["kinesthetic_token"] = kinesthetic_token_str
                    except Exception as e:
                        app.logger.error(
                            f"Error generating kinesthetic token: {str(e)}"
                        )

                    # Generate and store audio token in session for later use
                    try:
                        audio_token = auth.create_custom_token(user.uid, app=audio_app)
                        audio_token_str = audio_token.decode(
                            "utf-8"
                        )  # Convert bytes to string
                        session["audio_token"] = audio_token_str
                    except Exception as e:
                        app.logger.error(f"Error generating audio token: {str(e)}")

                    # Generate and store readwrite token in session for later use
                    try:
                        readwrite_token = auth.create_custom_token(
                            user.uid, app=readwrite_app
                        )
                        readwrite_token_str = readwrite_token.decode(
                            "utf-8"
                        )  # Convert bytes to string
                        session["readwrite_token"] = readwrite_token_str
                    except Exception as e:
                        app.logger.error(f"Error generating readwrite token: {str(e)}")

                    # Redirect to student dashboard instead of kinesthetic system
                    return redirect(url_for("student_dashboard"))
                else:
                    flash("Invalid user type")
                    return render_template("login.html")

            except Exception as e:
                flash(f"Login failed: {str(e)}")
                return render_template("login.html")

        return render_template("login.html")

    @app.route("/student_dashboard")
    @login_required
    def student_dashboard():
        """Student dashboard."""
        user_id = session.get("user_id")
        user_type = session.get("user_type")

        # Ensure user is a student
        if user_type != "student":
            flash("You need to be logged in as a student to access this page")
            return redirect(url_for("login"))

        # Get student data from Firestore
        student_data = Student.get(user_id)
        if not student_data:
            flash("Student data not found")
            return redirect(url_for("logout"))

        # Get teacher data for this student
        teacher_id = student_data.get("teacher_id")
        teacher_data = Teacher.get(teacher_id) if teacher_id else None

        # Get kinesthetic token from session or generate a new one if needed
        kinesthetic_token = session.get("kinesthetic_token")
        if not kinesthetic_token:
            try:
                custom_token = auth.create_custom_token(user_id, app=kinesthetic_app)
                kinesthetic_token = custom_token.decode("utf-8")
                session["kinesthetic_token"] = kinesthetic_token
            except Exception as e:
                app.logger.error(f"Error generating kinesthetic token: {str(e)}")

        # Get audio token from session or generate a new one if needed
        audio_token = session.get("audio_token")
        if not audio_token:
            try:
                custom_token = auth.create_custom_token(user_id, app=audio_app)
                audio_token = custom_token.decode("utf-8")
                session["audio_token"] = audio_token
            except Exception as e:
                app.logger.error(f"Error generating audio token: {str(e)}")

        # Get readwrite token from session or generate a new one if needed
        readwrite_token = session.get("readwrite_token")
        if not readwrite_token:
            try:
                custom_token = auth.create_custom_token(user_id, app=readwrite_app)
                readwrite_token = custom_token.decode("utf-8")
                session["readwrite_token"] = readwrite_token
            except Exception as e:
                app.logger.error(f"Error generating readwrite token: {str(e)}")

        # Create full URLs with tokens for apps
        kinesthetic_url = (
            f"http://localhost:{config.KINESTHETIC_APP_PORT}/authenticate?token={kinesthetic_token}"
            if kinesthetic_token
            else f"http://localhost:{config.KINESTHETIC_APP_PORT}"
        )
        audio_url = (
            f"http://localhost:{config.AUDIO_APP_PORT}/authenticate?token={audio_token}"
            if audio_token
            else f"http://localhost:{config.AUDIO_APP_PORT}"
        )
        readwrite_url = (
            f"http://localhost:{config.READWRITE_APP_PORT}/authenticate?token={readwrite_token}"
            if readwrite_token
            else f"http://localhost:{config.READWRITE_APP_PORT}"
        )

        return render_template(
            "student_dashboard.html",
            student=student_data,
            teacher=teacher_data,
            kinesthetic_url=kinesthetic_url,
            readwrite_url=readwrite_url,
            visual_url=f"http://localhost:{config.VISUAL_APP_PORT}",
            audio_url=audio_url,
        )

    # Add API endpoint for user details
    @app.route("/api/user/<user_id>", methods=["GET"])
    def get_user(user_id):
        try:
            # Check for authorization token (in a real app, validate this properly)
            # auth_header = request.headers.get('Authorization')
            # if not auth_header:
            #     return jsonify({'error': 'Authorization required'}), 401

            student = Student.get(user_id)
            if not student:
                return jsonify({"error": "Student not found"}), 404

            return jsonify(
                {
                    "id": student["id"],
                    "name": student.get("name", ""),
                    "email": student.get("email", ""),
                    "gender": student.get("gender", ""),
                    "birthday": student.get("birthday", ""),
                    "grade": student.get("grade", ""),
                }
            )
        except Exception as e:
            app.logger.error(f"Error getting user data: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Add new API endpoint for user progress
    @app.route("/api/user/<user_id>/progress", methods=["GET"])
    def get_user_progress(user_id):
        try:
            # Get Firestore client
            db = firestore.client()

            # Query the kinesthetic_marks collection for this user's marks
            marks_ref = (
                db.collection("kinesthetic_marks").where("user_id", "==", user_id).get()
            )

            # Calculate total score and count unique quiz attempts
            total_score = 0
            completed_quiz_ids = set()

            for mark in marks_ref:
                mark_data = mark.to_dict()
                score = mark_data.get("score", 0)
                quiz_id = mark_data.get("quiz_id")

                total_score += score
                if quiz_id:  # Only count if quiz_id exists
                    completed_quiz_ids.add(quiz_id)

            # Return the progress data as JSON
            return jsonify(
                {
                    "total_score": total_score,
                    "questions_completed": len(completed_quiz_ids),
                }
            )

        except Exception as e:
            app.logger.error(f"Error getting user progress: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Add API endpoint to save marks
    @app.route("/api/save_marks", methods=["POST"])
    def save_marks():
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

            user_id = data.get("user_id")
            quiz_id = data.get("quiz_id")
            score = data.get("score")
            subject_data = data.get("subject", {})

            if not user_id or score is None:
                return jsonify({"error": "Missing user_id or score"}), 400

            # Get Firestore client
            db = firestore.client()

            # Store marks in the main system's Firestore
            db.collection("kinesthetic_marks").add(
                {
                    "user_id": user_id,
                    "quiz_id": quiz_id,
                    "score": score,
                    "subject_data": subject_data,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                }
            )

            return jsonify({"status": "success"})
        except Exception as e:
            app.logger.error(f"Error saving marks: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/register", methods=["GET", "POST"])
    def register():
        """Handle teacher registration."""
        if request.method == "POST":
            email = request.form.get("email")
            password = request.form.get("password")
            confirm_password = request.form.get("confirm_password")
            name = request.form.get("name")
            school = request.form.get("school")

            if (
                not email
                or not password
                or not confirm_password
                or not name
                or not school
            ):
                flash("Please fill in all fields")
                return render_template("register.html")

            if password != confirm_password:
                flash("Passwords do not match")
                return render_template("register.html")

            try:
                # Create user in Firebase Authentication
                user = User.create_user(email, password)

                # Store additional teacher data in Firestore
                Teacher.create(user.uid, name, email, school)

                flash("Registration successful! Please login.")
                return redirect(url_for("login"))

            except Exception as e:
                flash(f"Registration failed: {str(e)}")
                return render_template("register.html")

        return render_template("register.html")

    @app.route("/dashboard")
    @teacher_required
    def dashboard():
        """Teacher dashboard."""
        user_id = session.get("user_id")

        # Get teacher data from Firestore
        teacher_data = Teacher.get(user_id)

        # Get all students for this teacher
        students = Student.get_all_for_teacher(user_id)

        return render_template(
            "dashboard.html",
            teacher=teacher_data,
            students=students,
            kinesthetic_url=f"http://localhost:{config.KINESTHETIC_APP_PORT}",
            readwrite_url=f"http://localhost:{config.READWRITE_APP_PORT}",
            visual_url=f"http://localhost:{config.VISUAL_APP_PORT}",
            audio_url=f"http://localhost:{config.AUDIO_APP_PORT}",
        )

    @app.route("/add_student", methods=["POST"])
    @teacher_required
    def add_student():
        """Add a new student for the teacher."""
        user_id = session.get("user_id")

        name = request.form.get("student_name")
        email = request.form.get("student_email")
        password = request.form.get("student_password")
        gender = request.form.get("student_gender")
        birthday = request.form.get("student_birthday")
        grade = request.form.get("student_grade")

        if not name or not email or not password:
            flash("Please fill in all required student details")
            return redirect(url_for("dashboard"))

        try:
            # Create the student with new fields
            # Even if gender/birthday/grade are empty strings, pass them anyway
            Student.create(
                user_id,
                name,
                email,
                password,
                gender=gender if gender else None,
                birthday=birthday if birthday else None,
                grade=grade if grade else None,
            )
            flash(f"Student {name} has been added successfully")
        except Exception as e:
            flash(f"Failed to add student: {str(e)}")

        return redirect(url_for("dashboard"))

    @app.route("/edit_student/<student_id>", methods=["GET", "POST"])
    @teacher_required
    def edit_student(student_id):
        """Edit a student's details."""
        user_id = session.get("user_id")

        # Get student data
        student = Student.get(student_id)

        # Check if the student belongs to this teacher
        if not student or student.get("teacher_id") != user_id:
            flash("You do not have permission to edit this student")
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            name = request.form.get("student_name")
            email = request.form.get("student_email")
            gender = request.form.get("student_gender")
            birthday = request.form.get("student_birthday")
            grade = request.form.get("student_grade")

            # Debug - optional, remove in production
            app.logger.info(
                f"Edit student data - Gender: {gender}, Birthday: {birthday}, Grade: {grade}"
            )

            try:
                # Always pass the values to update, even if they're empty strings
                # The model will handle them appropriately
                Student.update(
                    student_id,
                    name=name if name else None,
                    email=email if email else None,
                    gender=gender if gender else None,
                    birthday=birthday if birthday else None,
                    grade=grade if grade else None,
                )

                # If it's an AJAX request, return a JSON response
                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return jsonify(
                        {
                            "success": True,
                            "message": f"Student {name} has been updated successfully",
                        }
                    )

                flash(f"Student {name} has been updated successfully")
                return redirect(url_for("dashboard"))

            except Exception as e:
                app.logger.error(f"Error updating student: {str(e)}")
                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return jsonify({"error": str(e)}), 500
                flash(f"Failed to update student: {str(e)}")
                return redirect(url_for("dashboard"))

        # GET request is not needed anymore since we're using a modal
        # Just redirect to dashboard if someone tries to access this URL directly
        return redirect(url_for("dashboard"))

    @app.route("/delete_student/<student_id>")
    @teacher_required
    def delete_student(student_id):
        """Delete a student."""
        user_id = session.get("user_id")

        # Get student data
        student = Student.get(student_id)

        # Check if the student belongs to this teacher
        if not student or student.get("teacher_id") != user_id:
            flash("You do not have permission to delete this student")
            return redirect(url_for("dashboard"))

        try:
            Student.delete(student_id)
            flash("Student has been deleted successfully")
        except Exception as e:
            flash(f"Failed to delete student: {str(e)}")

        return redirect(url_for("dashboard"))

    @app.route("/logout")
    def logout():
        """Log out the user."""
        # Clear the entire session instead of individual keys
        session.clear()
        flash("You have been logged out")
        # Explicit redirect to login page
        return redirect(url_for("login"))

    @app.route("/kinesthetic")
    def kinesthetic_redirect():
        """Redirect to the kinesthetic app."""
        return redirect(f"http://localhost:{config.KINESTHETIC_APP_PORT}")

    @app.route("/readwrite")
    def readwrite_redirect():
        """Redirect to the readwrite app."""
        return redirect(f"http://localhost:{config.READWRITE_APP_PORT}")

    @app.route("/visual")
    def visual_redirect():
        """Redirect to the visual app."""
        return redirect(f"http://localhost:{config.VISUAL_APP_PORT}")

    @app.route("/audio")
    def audio_redirect():
        """Redirect to the audio app."""
        return redirect(f"http://localhost:{config.AUDIO_APP_PORT}")

    @app.route("/api/status")
    def api_status():
        """API endpoint to check the status of all apps."""
        status = {
            "main": "running",
            "kinesthetic": "unknown",
            "readwrite": "unknown",
            "visual": "unknown",
            "audio": "unknown",
        }

        # Check kinesthetic app
        try:
            response = requests.get(
                f"http://localhost:{config.KINESTHETIC_APP_PORT}/api/info"
            )
            if response.status_code == 200:
                status["kinesthetic"] = "running"
        except:
            status["kinesthetic"] = "not running"

        # Check readwrite app
        try:
            response = requests.get(
                f"http://localhost:{config.READWRITE_APP_PORT}/api/info"
            )
            if response.status_code == 200:
                status["readwrite"] = "running"
        except:
            status["readwrite"] = "not running"

        # Check visual app
        try:
            response = requests.get(
                f"http://localhost:{config.VISUAL_APP_PORT}/api/info"
            )
            if response.status_code == 200:
                status["visual"] = "running"
        except:
            status["visual"] = "not running"

        # Check audio app
        try:
            response = requests.get(
                f"http://localhost:{config.AUDIO_APP_PORT}/api/info"
            )
            if response.status_code == 200:
                status["audio"] = "running"
        except:
            status["audio"] = "not running"

        return jsonify(status)

    @app.route("/api/system_metrics")
    def system_metrics():
        """API endpoint to get real-time system metrics."""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Get system uptime
            boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.datetime.now() - boot_time
            uptime_hours = round(uptime.total_seconds() / 3600, 1)  # Convert to hours

            # Get number of active processes as a proxy for "users"
            active_processes = len(psutil.pids())

            # Get historical CPU and memory data for chart (last 10 minutes)
            # In a real implementation, you would store and retrieve this from a time-series database
            # For now, we'll just return the current value repeated
            cpu_history = [cpu_percent] * 10
            memory_history = [memory_percent] * 10

            # Get active users by time period (mock data, would be from database in real app)
            user_activity = [active_processes // 5] * 6  # Divide by 5 for demonstration

            return jsonify(
                {
                    "cpu": {"current": cpu_percent, "history": cpu_history},
                    "memory": {"current": memory_percent, "history": memory_history},
                    "uptime": {
                        "hours": uptime_hours,
                        "formatted": f"{int(uptime_hours)}h {int((uptime_hours % 1) * 60)}m",
                    },
                    "active_processes": active_processes,
                    "user_activity": user_activity,
                }
            )
        except Exception as e:
            app.logger.error(f"Error getting system metrics: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/system_overview")
    def system_overview():
        """System overview page showing status of all applications."""
        user_id = session.get("user_id")
        user_type = session.get("user_type")

        return render_template("system_overview.html", user_type=user_type)

    # Ensure this function is registered, debug it
    app.logger.info("Registered route: /system_overview")

    @app.route("/classify_students", methods=["GET", "POST"])
    @teacher_required
    def classify_students():
        """Classify students using the VARK model."""
        user_id = session.get("user_id")

        if request.method == "POST":
            # Use the specified teacher ID for testing purposes
            teacher_id = (
                "uPSpxGSRFYdnexxEDh45TxUznRJ3"  # Using the fixed ID as requested
            )

            try:
                # Step 1: Retrieve student data using the function from retrieve_students.py
                retrieve_students_data(teacher_id=teacher_id)

                # The file will be saved as teacher-{teacher_id}-students.json
                students_file = f"teacher-{teacher_id}-students.json"

                if not os.path.exists(students_file):
                    flash(f"No student data found for teacher ID: {teacher_id}")
                    return redirect(url_for("dashboard"))

                # Step 2: Run the classification process
                original_argv = sys.argv.copy()
                sys.argv = [
                    "classify_students.py",
                    "--teacher",
                    teacher_id,
                    "--input",
                    students_file,
                ]

                try:
                    classify_main()
                finally:
                    # Restore original sys.argv
                    sys.argv = original_argv

                # Step 3: Load the classification results
                results_file = "vark_results.json"
                if not os.path.exists(results_file):
                    flash("Classification process did not generate results")
                    return redirect(url_for("dashboard"))

                with open(results_file, "r") as f:
                    classification_results = json.load(f)

                # No need for this block anymore since we're adding names directly in the
                # classifier, but keep it as a fallback in case any names are missing
                # Load student data to add any missing names
                with open(students_file, "r") as f:
                    students_data = json.load(f)

                # Create a dictionary mapping student IDs to names
                student_names = {}
                for student in students_data:
                    student_id = student.get("id")
                    if student_id:
                        student_names[student_id] = student.get("name", "Unknown")

                # Fill in any missing student names
                for student in classification_results["classifications"]:
                    if (
                        not student.get("student_name")
                        and student["student_id"] in student_names
                    ):
                        student["student_name"] = student_names[student["student_id"]]

                # Step 4: Load the visualization images
                visualization_files = [
                    "learning_styles_bar.png",
                    "score_distributions_box.png",
                    "threshold_comparison_violin.png",
                    "modality_correlations.png",
                    "learning_style_radar.png",
                    "learning_style_pie.png",
                    "modality_distributions.png",
                ]

                # Check if the visualization directory exists
                visualizations_dir = "visualizations"

                # Verify each file exists
                available_visualizations = []
                for viz_file in visualization_files:
                    viz_path = os.path.join(visualizations_dir, viz_file)
                    if os.path.exists(viz_path):
                        available_visualizations.append(viz_file)

                # Load analysis text files
                analysis_text = {}
                for text_file in ["analysis_summary.txt", "statistical_analysis.txt"]:
                    text_path = os.path.join(visualizations_dir, text_file)
                    if os.path.exists(text_path):
                        with open(text_path, "r") as f:
                            analysis_text[text_file] = f.read()

                # Return the classification results template
                return render_template(
                    "classify_results.html",
                    results=classification_results,
                    visualizations=available_visualizations,
                    analysis_text=analysis_text,
                )

            except Exception as e:
                app.logger.error(f"Error in classification process: {str(e)}")
                flash(f"Error during classification: {str(e)}")
                return redirect(url_for("dashboard"))

        # If GET request, just return to dashboard
        return redirect(url_for("dashboard"))

    @app.route("/visualizations/<path:filename>")
    def visualization_file(filename):
        """Serve visualization files."""
        return send_from_directory("visualizations", filename)

    # Ensure this function is registered, debug it
    app.logger.info("Registered route: /system_overview")

    @app.route("/api/student/<student_id>/progress")
    @teacher_required
    def student_progress(student_id):
        """Get detailed progress data for a specific student."""
        try:
            # Get student info
            student = Student.get(student_id)
            if not student:
                return jsonify({"error": "Student not found"}), 404

            # Get Firestore client
            db = firestore.client()

            # Get student's kinesthetic marks/quiz results
            marks_ref = (
                db.collection("kinesthetic_marks")
                .where("user_id", "==", student_id)
                .get()
            )

            # Process the marks data
            total_score = 0
            completed_quiz_ids = set()
            subject_progress = {
                "addition": 0,
                "subtraction": 0,
                "time": 0,
                "measurement": 0,
                "geometry": 0,
                "overall": 0,
            }

            quiz_attempts = []
            recent_activities = []

            # Process each mark/quiz result
            for mark in marks_ref:
                mark_data = mark.to_dict()
                score = mark_data.get("score", 0)
                quiz_id = mark_data.get("quiz_id", "unknown")
                timestamp = mark_data.get("timestamp")
                subject_data = mark_data.get("subject_data", {})

                # Calculate totals
                total_score += score
                if quiz_id:
                    completed_quiz_ids.add(quiz_id)

                # Calculate subject progress
                if subject_data:
                    subject = subject_data.get("subject", "").lower()
                    if subject in subject_progress:
                        # Add to the subject's score
                        subject_progress[subject] += score

                # Add to quiz attempts
                quiz_attempts.append(
                    {
                        "quiz_id": quiz_id,
                        "score": score,
                        "timestamp": timestamp.isoformat() if timestamp else None,
                        "subject": subject_data.get("subject", "Unknown"),
                    }
                )

                # Add to recent activities
                recent_activities.append(
                    {
                        "type": "quiz",
                        "details": f"Completed {subject_data.get('subject', 'Quiz')}",
                        "score": score,
                        "timestamp": timestamp.isoformat() if timestamp else None,
                    }
                )

            # Normalize subject progress to percentages (mock calculation)
            for subject in subject_progress:
                # Simple mock calculation for demo purposes
                max_subject_score = 100  # Theoretical maximum
                subject_score = min(subject_progress[subject], max_subject_score)
                subject_progress[subject] = int(
                    (subject_score / max_subject_score) * 100
                )

            # Calculate overall progress
            if subject_progress:
                non_zero_subjects = [
                    score for score in subject_progress.values() if score > 0
                ]
                subject_progress["overall"] = int(
                    sum(non_zero_subjects) / max(len(non_zero_subjects), 1)
                )

            # Get VARK learning style data if available
            learning_style = "Unknown"
            vark_scores = None

            # Try to find the most recent VARK classification
            try:
                # Check if there's a vark_results.json file
                if os.path.exists("vark_results.json"):
                    with open("vark_results.json", "r") as f:
                        vark_data = json.load(f)

                    for classification in vark_data.get("classifications", []):
                        if classification.get("student_id") == student_id:
                            learning_style = classification.get(
                                "learning_style", "Unknown"
                            )
                            vark_scores = classification.get("scores")
                            break
            except Exception as e:
                app.logger.error(f"Error getting learning style: {e}")

            # Sort recent activities by timestamp
            recent_activities.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return jsonify(
                {
                    "student_id": student_id,
                    "student_name": student.get("name", "Unknown"),
                    "total_score": total_score,
                    "questions_completed": len(completed_quiz_ids),
                    "subject_progress": subject_progress,
                    "quiz_attempts": quiz_attempts[:10],  # Limit to most recent 10
                    "recent_activities": recent_activities[
                        :5
                    ],  # Limit to most recent 5
                    "learning_style": learning_style,
                    "vark_scores": vark_scores,
                    "grade": student.get("grade", "Unknown"),
                    "last_active": (
                        recent_activities[0].get("timestamp")
                        if recent_activities
                        else None
                    ),
                }
            )

        except Exception as e:
            app.logger.error(f"Error getting student progress: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/student/<student_id>/learning_style", methods=["POST"])
    def update_student_learning_style(student_id):
        """API endpoint to update a student's learning style scores."""
        try:
            # Get the data from request
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

            # Extract style and score from request
            style = data.get("style")
            score = data.get("score")

            # Validate inputs
            if not style or score is None:
                return jsonify({"error": "Missing style or score"}), 400

            # Validate style is one of the allowed values
            if style not in ["reading", "kinesthetic", "visual", "auditory"]:
                return (
                    jsonify(
                        {
                            "error": "Invalid learning style. Must be one of: reading, kinesthetic, visual, auditory"
                        }
                    ),
                    400,
                )

            # Validate score is a number
            try:
                score = int(score)
            except ValueError:
                return jsonify({"error": "Score must be a number"}), 400

            # Update the student's learning style
            Student.update_learning_style(student_id, style, score)

            return jsonify(
                {
                    "success": True,
                    "message": f"Successfully updated {style} learning style score to {score}",
                    "student_id": student_id,
                }
            )

        except Exception as e:
            app.logger.error(f"Error updating learning style: {str(e)}")
            return jsonify({"error": str(e)}), 500

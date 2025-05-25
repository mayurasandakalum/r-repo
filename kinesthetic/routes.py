@kinesthetic_blueprint.route("/user-home")
@login_required
def user_home():
    kinesthetic_profile = QuizProfile.get_by_user_id(current_user.id)
    
    # Get attempts by quiz type for performance comparison
    attempts_by_type = {}
    weakest_subject = None
    
    if kinesthetic_profile:
        # Prioritize the stored weakest_subject instead of recalculating
        weakest_subject = kinesthetic_profile.weakest_subject
        
        # Only calculate if no stored value exists (for backward compatibility)
        if not weakest_subject and kinesthetic_profile.subject_performance:
            weakest_data = kinesthetic_profile.get_weakest_subject()
            weakest_subject = weakest_data.get("subject")
        
        # Only fetch attempts if we have a weakest subject and no stored comparison data
        if weakest_subject and not (hasattr(kinesthetic_profile, 'quiz_comparisons') and 
                                    weakest_subject in kinesthetic_profile.quiz_comparisons):
            # Get attempts specifically for this subject, not all attempts
            attempts_query = (
                db.collection("attempted_questions")
                .where("user_id", "==", current_user.id)
                .where("quiz_type", "in", ["mixed_quiz", "weakest_subject"])
                .get()
            )
            
            # Collect all question IDs first to batch fetch them
            question_ids = []
            attempt_by_question = {}
            for attempt in attempts_query:
                attempt_data = attempt.to_dict()
                question_id = attempt_data.get("question_id")
                if question_id:
                    question_ids.append(question_id)
                    if question_id not in attempt_by_question:
                        attempt_by_question[question_id] = []
                    attempt_by_question[question_id].append(attempt_data)
            
            # Batch get questions (in chunks of 10 to avoid Firestore limits)
            question_data_map = {}
            for i in range(0, len(question_ids), 10):
                batch_ids = question_ids[i:i+10]
                questions_batch = db.collection("questions").where(firestore.field_path.FieldPath.document_id(), "in", batch_ids).get()
                for q_doc in questions_batch:
                    question_data_map[q_doc.id] = q_doc.to_dict()
            
            # Now process attempts with the question data we already have
            for question_id, attempts in attempt_by_question.items():
                if question_id not in question_data_map:
                    continue
                    
                question_data = question_data_map[question_id]
                subject = question_data.get("subject")
                
                if subject != weakest_subject:
                    continue
                
                for attempt_data in attempts:
                    quiz_type = attempt_data.get("quiz_type", "mixed_quiz")
                    
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
        weakest_subject=weakest_subject
    )

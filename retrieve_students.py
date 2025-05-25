import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import os
import datetime

# Custom JSON encoder to handle Firebase types
class FirestoreEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle Firestore timestamp objects
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        # Handle any other Firebase special types if needed
        return super().default(obj)

# Move Firebase initialization inside the function to avoid conflicts
# when this module is imported by routes.py
def retrieve_students_data(teacher_id="uPSpxGSRFYdnexxEDh45TxUznRJ3"):
    # Initialize Firebase only if needed
    db = None
    app = None
    
    try:
        # Check if Firebase is already initialized
        try:
            # Try to get an existing app
            app = firebase_admin.get_app()
            db = firestore.client()
            print("Using existing Firebase app")
        except ValueError:
            # No default app exists, so initialize one with a name
            cred = credentials.Certificate("research-app-9fff9-firebase-adminsdk-fbsvc-35cdf97b1e.json")
            app = firebase_admin.initialize_app(cred, name='student_retrieval')
            db = firestore.client(app=app)
            print("Initialized new Firebase app for student retrieval")
        
        # First, get ALL students to analyze the data structure
        all_students_ref = db.collection("students")
        all_snapshot = all_students_ref.get()
        
        if not all_snapshot:
            print("No students found in the collection at all.")
            return
            
        # Look for teacher-related fields in the student documents
        print(f"Analyzing {len(all_snapshot)} student documents to find teacher relationship...")
        teacher_field_names = set()
        has_target_teacher = False
        
        for doc in all_snapshot:
            student = doc.to_dict()
            # Print all fields in the first document to understand structure
            if len(teacher_field_names) == 0:
                print(f"Available fields in student document: {list(student.keys())}")
            
            # Look for any field that might contain our teacher ID
            for field, value in student.items():
                if 'teacher' in field.lower():
                    teacher_field_names.add(field)
                if value == teacher_id:
                    print(f"Found teacher ID in field: {field}")
                    has_target_teacher = True
        
        print(f"Possible teacher-related fields found: {list(teacher_field_names)}")
        if not has_target_teacher:
            print(f"Warning: Teacher ID '{teacher_id}' was not found in any student document")
        
        # Now try different possibilities for the teacher field
        students_data = []
        for field_name in teacher_field_names:
            print(f"Trying to query with field: {field_name}")
            filtered_ref = db.collection("students").where(field_name, "==", teacher_id)
            snapshot = filtered_ref.get()
            
            if snapshot:
                print(f"Found {len(snapshot)} students using field {field_name}")
                for doc in snapshot:
                    student = doc.to_dict()
                    student["id"] = doc.id
                    students_data.append(student)
        
        # If no matches with teacher fields, try a last resort approach
        if not students_data:
            # Look for classes/sections data
            classes_ref = db.collection("classes")
            if classes_ref:
                classes_snapshot = classes_ref.where("teacherId", "==", teacher_id).get()
                if classes_snapshot:
                    print(f"Found {len(classes_snapshot)} classes for this teacher")
                    # Get student IDs from these classes
                    for class_doc in classes_snapshot:
                        class_data = class_doc.to_dict()
                        if "studentIds" in class_data and class_data["studentIds"]:
                            print(f"Found {len(class_data['studentIds'])} students in class {class_doc.id}")
                            for student_id in class_data["studentIds"]:
                                student_doc = db.collection("students").document(student_id).get()
                                if student_doc.exists:
                                    student = student_doc.to_dict()
                                    student["id"] = student_doc.id
                                    students_data.append(student)
        
        if not students_data:
            print(f"Could not find any students related to teacher ID: {teacher_id}")
            return
        
        # Convert to formatted JSON using custom encoder
        json_data = json.dumps(students_data, indent=2, cls=FirestoreEncoder)
        
        # Define output file path with teacher ID
        output_path = os.path.join(os.path.dirname(__file__), f"teacher-{teacher_id}-students.json")
        
        # Write data to file
        with open(output_path, "w") as f:
            f.write(json_data)
        
        print(f"Successfully retrieved {len(students_data)} student records for teacher ID: {teacher_id}")
        print(f"Data saved to: {output_path}")
        
    except Exception as error:
        print(f"Error retrieving student data: {error}")
    finally:
        # Don't delete apps that existed before this function was called
        if app and app.name == 'student_retrieval':
            try:
                firebase_admin.delete_app(app)
            except:
                pass

# Execute the function only if this script is run directly
if __name__ == "__main__":
    retrieve_students_data()

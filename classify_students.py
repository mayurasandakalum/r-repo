import argparse
import json
import os
import sys
from vark_classifier import VARKClassifier
from retrieve_students import retrieve_students_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Classify students' learning styles using VARK")
    parser.add_argument("--teacher", default="uPSpxGSRFYdnexxEDh45TxUznRJ3", 
                      help="Teacher ID to retrieve students for")
    parser.add_argument("--input", help="Path to existing student data JSON file (skips Firebase retrieval)")
    parser.add_argument("--output", default="vark_results.json", 
                      help="Output file path for classification results")
    args = parser.parse_args()
    
    # Step 1: Get student data
    if args.input:
        # Use provided file
        students_file = args.input
        print(f"Using provided student data file: {students_file}")
    else:
        # Retrieve from Firebase
        print(f"Retrieving student data for teacher ID: {args.teacher}")
        retrieve_students_data(args.teacher)
        students_file = f"teacher-{args.teacher}-students.json"
    
    # Check if file exists
    if not os.path.exists(students_file):
        print(f"Error: Student data file {students_file} not found.")
        sys.exit(1)
    
    # Step 2: Transform data for VARK classifier
    try:
        # Read the student data
        with open(students_file, "r") as f:
            firebase_students = json.load(f)
        
        # Transform data
        vark_data = {"students": []}
        
        for student in firebase_students:
            # Check if student has VARK scores (stored in learning_styles field)
            if "learning_styles" not in student or not student["learning_styles"]:
                print(f"Warning: Student {student.get('name', student.get('id', 'unknown'))} missing learning styles")
                continue
                
            vark_scores = student["learning_styles"]
            
            # Check if learning_styles has the required fields
            required_fields = ["visual", "auditory", "reading", "kinesthetic"]
            if not all(key in vark_scores for key in required_fields):
                missing = [field for field in required_fields if field not in vark_scores]
                print(f"Warning: Student {student.get('name', student.get('id', 'unknown'))} missing fields: {', '.join(missing)}")
                continue
            
            # Ensure values are numeric
            try:
                vark_student = {
                    "id": student["id"],
                    "name": student.get("name", student["id"]),  # Include the name
                    "visual": float(vark_scores["visual"]),
                    "auditory": float(vark_scores["auditory"]),
                    "reading": float(vark_scores["reading"]),
                    "kinesthetic": float(vark_scores["kinesthetic"])
                }
                vark_data["students"].append(vark_student)
            except (ValueError, TypeError) as e:
                print(f"Warning: Student {student.get('name', student.get('id', 'unknown'))} has non-numeric VARK scores: {e}")
        
        if not vark_data["students"]:
            print("Error: No valid student VARK data found.")
            sys.exit(1)
            
        print(f"Found {len(vark_data['students'])} students with valid VARK scores")
        
        # Save transformed data for VARK classifier
        vark_input_file = "vark_input.json"
        with open(vark_input_file, "w") as f:
            json.dump(vark_data, f, indent=2)
        
    except Exception as e:
        print(f"Error processing student data: {e}")
        sys.exit(1)
    
    # Step 3: Run VARK classifier
    try:
        print("Running VARK classification...")
        classifier = VARKClassifier()
        classifier.process_data(vark_input_file, args.output)
        
        print(f"Classification complete. Results saved to {args.output}")
        
        # Step 4: Show a summary of results
        with open(args.output, "r") as f:
            results = json.load(f)
        
        print("\n===== VARK Classification Summary =====")
        print(f"Total students classified: {len(results['classifications'])}")
        
        # Count learning styles
        learning_styles = {}
        for classification in results['classifications']:
            style = classification["learning_style"]
            if style in learning_styles:
                learning_styles[style] += 1
            else:
                learning_styles[style] = 1
        
        print("\nLearning Style Distribution:")
        for style, count in sorted(learning_styles.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results['classifications'])) * 100
            print(f"{style}: {count} students ({percentage:.1f}%)")
        
        print("\nThresholds used:")
        for modality, threshold in results["thresholds"].items():
            print(f"{modality}: {threshold:.2f}")
            
        print("\nVisualization files generated:")
        visualization_files = [
            "learning_styles_bar.png", "score_distributions_box.png", 
            "threshold_comparison_violin.png", "modality_correlations.png",
            "learning_style_radar.png", "learning_style_pie.png",
            "modality_distributions.png", "analysis_summary.txt",
            "statistical_analysis.txt"
        ]
        for viz_file in visualization_files:
            if os.path.exists(viz_file):
                print(f"- {viz_file}")
            
    except Exception as e:
        print(f"Error running VARK classification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

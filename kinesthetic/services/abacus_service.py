import base64
import os
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from pathlib import Path

# Initialize the model once
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "models", "abacus-model", "model-files", "best-3.pt")
print("\n[INIT 1] Loading abacus model from: {0}".format(MODEL_PATH))
abacus_model = YOLO(MODEL_PATH)
print("[INIT 2] Abacus model loaded successfully\n")

def decode_base64_image(base64_string):
    """Convert base64 string to OpenCV image"""
    print("[IMAGE 1] Decoding base64 image for abacus processing")
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    image_bytes = base64.b64decode(base64_string)
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    print("[IMAGE 2] Image decoded: shape={0}".format(image.shape if image is not None else 'None'))
    return image

def save_base64_image(base64_string, prefix="abacus"):
    """Save base64 image to a temporary file and return the path"""
    print("[SAVE 1] Saving base64 image to temporary file")
    image = decode_base64_image(base64_string)
    
    # Create temporary file with .jpg extension
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".jpg", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Save image to the temporary file
    cv2.imwrite(temp_path, image)
    print("[SAVE 2] Image saved to temporary file: {0}".format(temp_path))
    return temp_path

def process_abacus_image(base64_image):
    """Process an abacus image and return the detected value"""
    print("\n[PROCESS 1] ===== STARTING ABACUS IMAGE PROCESSING =====")
    # Save the base64 image to a file
    image_path = save_base64_image(base64_image)
    
    try:
        # Run model inference
        print("\n[PROCESS 2] Running YOLO model inference on abacus image")
        results = abacus_model.predict(image_path)
        detections = results[0]
        print("[PROCESS 3] Model detected {0} objects".format(len(detections.boxes)))
        
        # Define class IDs
        BEAD_CLASS_ID = 0
        STICK_CLASS_ID = 1
        
        beads = []
        sticks = []
        
        # Extract bounding boxes for beads and sticks
        print("\n[DETECT 1] Extracting bounding boxes for beads and sticks")
        for result in detections.boxes:
            class_id = int(result.cls)
            bbox = result.xyxy.cpu().numpy()[0]
            if class_id == BEAD_CLASS_ID:
                beads.append(bbox)
            elif class_id == STICK_CLASS_ID:
                sticks.append(bbox)
        
        print("[DETECT 2] Extracted {0} beads and {1} sticks".format(len(beads), len(sticks)))
        
        # Sort sticks by x1 coordinate (from left to right)
        sticks = sorted(sticks, key=lambda bbox: bbox[0])
        print("[DETECT 3] Sorted sticks from left to right")
        
        # Assign beads to sticks
        print("\n[COUNT 1] Assigning beads to sticks")
        stick_bead_counts = {}
        for idx, stick_bbox in enumerate(sticks):
            x1s, y1s, x2s, y2s = stick_bbox
            count = 0
            for bead_bbox in beads:
                x1b, y1b, x2b, y2b = bead_bbox
                cx = (x1b + x2b) / 2
                cy = (y1b + y2b) / 2
                if x1s <= cx <= x2s and y1s <= cy <= y2s:
                    count += 1
            stick_bead_counts[idx + 1] = count  # Sticks numbered from 1
            print("[COUNT 2] Stick {0} has {1} beads".format(idx+1, count))
        
        # Calculate total abacus value
        print("\n[CALC 1] Calculating total abacus value")
        stick_values = {1: 10000, 2: 1000, 3: 100, 4: 10, 5: 1}
        total_value = sum(
            bead_count * stick_values.get(stick_num, 0)
            for stick_num, bead_count in stick_bead_counts.items()
        )
        print("[CALC 2] Total abacus value: {0}".format(total_value))
        
        # Create annotated image path
        base_path = Path(image_path)
        annotated_path = base_path.parent / f"annotated_{base_path.name}"
        
        # Create an annotator and draw boxes
        print("\n[ANNOTATE 1] Creating annotated image with detections")
        original_img = cv2.imread(image_path)
        from ultralytics.utils.plotting import Annotator
        annotator = Annotator(original_img, line_width=1, font_size=0.5)
        
        # Define class names
        class_names = {0: "bead", 1: "stick"}
        
        # Draw detections
        for result in detections.boxes:
            class_id = int(result.cls)
            bbox = result.xyxy.cpu().numpy()[0]
            conf = float(result.conf)
            
            # Define colors based on class
            if class_id == BEAD_CLASS_ID:
                box_color = (255, 0, 0)  # Blue for beads (BGR)
                txt_color = (0, 255, 255)  # Yellow text (BGR)
            else:  # STICK_CLASS_ID
                box_color = (0, 255, 0)  # Green for sticks (BGR)
                txt_color = (255, 255, 255)  # White text (BGR)
                
            # Draw box and label
            label = f"{class_names.get(class_id, 'unknown')} {conf:.2f}"
            annotator.box_label(bbox, label, color=box_color, txt_color=txt_color)
        
        # Get and save annotated image
        annotated_img = annotator.result()
        cv2.imwrite(str(annotated_path), annotated_img)
        print("[ANNOTATE 2] Annotated image saved to: {0}".format(annotated_path))
        
        result = {
            'total_value': total_value,
            'bead_counts': list(sorted(stick_bead_counts.items())),
            'original_path': image_path,
            'annotated_path': str(annotated_path)
        }
        print("\n[COMPLETE] Abacus processing complete: {0}\n".format(result))
        return result
    
    finally:
        # In a production environment, you might want to clean up temporary files
        # os.unlink(image_path)
        pass

def check_abacus_answer(base64_image, expected_answer):
    """
    Check if the abacus image shows the expected answer
    Returns a tuple: (is_correct, detected_value, annotated_image_path)
    """
    print("\n" + "="*50)
    print("[CHECK 1] CHECKING ABACUS ANSWER (Expected: {0})".format(expected_answer))
    print("="*50)
    
    try:
        # Convert expected_answer to int for comparison
        expected_value = int(expected_answer)
        print("[CHECK 2] Converted expected answer to integer: {0}".format(expected_value))
        
        # Process the image
        print("\n[CHECK 3] Processing abacus image...")
        result = process_abacus_image(base64_image)
        detected_value = result['total_value']
        annotated_path = result['annotated_path']
        print("\n[CHECK 4] Processing complete. Detected value: {0}".format(detected_value))
        
        # Check if the detected value matches the expected value
        is_correct = (detected_value == expected_value)
        print("\n[RESULT] Answer correct? {0} (Detected: {1}, Expected: {2})".format(
            is_correct, detected_value, expected_value))
        print("="*50 + "\n")
        
        return is_correct, detected_value, annotated_path
    
    except Exception as e:
        # Log the error and return False
        print("\n[ERROR] Error checking abacus answer: {0}".format(str(e)))
        print("="*50 + "\n")
        return False, None, None

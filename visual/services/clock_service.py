import base64
import os
import cv2
import numpy as np
import torch
import einops
import torch.nn as nn
import torchvision.models as models
import tempfile
from pathlib import Path

# Use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n[INIT 1] Using device: {0} for clock detection model".format(device))

# Define the models
print("[INIT 2] Initializing clock models...")
model_stn = models.resnet50(pretrained=False)
model_stn.fc = nn.Linear(2048, 8)
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 720)

# Get the absolute path to the model files
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_stn_path = os.path.join(project_root, "models", "clock-model", "model-files", "model_stn.pth")
model_path = os.path.join(project_root, "models", "clock-model", "model-files", "model.pth")

print("[INIT 3] Loading STN model from: {0}".format(model_stn_path))
print("[INIT 4] Loading main model from: {0}".format(model_path))

# Load the saved model weights
model_stn.load_state_dict(torch.load(model_stn_path, map_location=device))
model.load_state_dict(torch.load(model_path, map_location=device))

model_stn.to(device)
model.to(device)
model_stn.eval()
model.eval()
print("[INIT 5] Clock models loaded and ready for inference\n")

# Define warp function (from original clock_app.py)
def warp(img, M):
    print("[WARP 1] Applying spatial transformer network warp")
    grid = nn.functional.affine_grid(M[:, :2], img.size(), align_corners=False)
    img_warped = nn.functional.grid_sample(img, grid, align_corners=False)
    print("[WARP 2] Warp complete")
    return img_warped

def decode_base64_image(base64_string):
    """Convert base64 string to OpenCV image"""
    print("[IMAGE 1] Decoding base64 image for clock processing")
    
    try:
        # Validate that we have some data
        if not base64_string or len(base64_string) < 10:
            print("[IMAGE ERROR] Base64 string is empty or too short")
            return None
            
        # Strip the prefix if it exists
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        # Clean the base64 string by removing invalid characters
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
        invalid_chars = []
        
        for i, char in enumerate(base64_string[:100]):
            if char not in valid_chars:
                invalid_chars.append(char)
        
        if invalid_chars:
            print(f"[IMAGE WARNING] Found invalid characters in base64 string: {invalid_chars}")
            clean_base64 = ''.join(c for c in base64_string if c in valid_chars)
            base64_string = clean_base64
            print(f"[IMAGE WARNING] Cleaned base64 string, length: {len(base64_string)}")
        
        # Add padding if needed (base64 requires length to be multiple of 4)
        padding_needed = len(base64_string) % 4
        if padding_needed > 0:
            print(f"[IMAGE 1.1] Fixing base64 padding (adding {4 - padding_needed} characters)")
            base64_string += '=' * (4 - padding_needed)
        
        # Decode the base64 data with better error handling
        try:
            image_bytes = base64.b64decode(base64_string)
            if not image_bytes or len(image_bytes) < 100:  # Basic validation
                print(f"[IMAGE ERROR] Decoded image bytes too small: {len(image_bytes) if image_bytes else 0} bytes")
                return None
                
            np_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print("[IMAGE ERROR] OpenCV could not decode the image data")
                return None
                
            print("[IMAGE 2] Image decoded successfully: shape={0}".format(image.shape))
            return image
            
        except base64.binascii.Error as e:
            print(f"[IMAGE ERROR] Base64 decoding error: {str(e)}")
            
            # One more try with aggressive cleaning
            try:
                # Try with more aggressive cleaning - keep only alphanumeric, +, /, and =
                import re
                cleaned = re.sub(r'[^A-Za-z0-9+/=]', '', base64_string)
                padding_needed = len(cleaned) % 4
                if padding_needed > 0:
                    cleaned += '=' * (4 - padding_needed)
                    
                print(f"[IMAGE RETRY] Attempting with aggressively cleaned base64, length: {len(cleaned)}")
                image_bytes = base64.b64decode(cleaned)
                np_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                
                if image is not None:
                    print("[IMAGE 2] Image decoded successfully after cleaning: shape={0}".format(image.shape))
                    return image
                else:
                    print("[IMAGE ERROR] OpenCV could not decode the image data even after cleaning")
                    return None
            except Exception as inner_e:
                print(f"[IMAGE ERROR] Failed after aggressive cleaning: {str(inner_e)}")
                return None
            
    except Exception as e:
        print(f"[IMAGE ERROR] Failed to decode base64 image: {str(e)}")
        return None

def save_base64_image(base64_string, prefix="clock"):
    """Save base64 image to a temporary file and return the path"""
    print("[SAVE 1] Saving base64 image to temporary file")
    
    # Decode the image
    image = decode_base64_image(base64_string)
    
    if image is None:
        print("[SAVE ERROR] Could not decode image from base64 data")
        # Create a placeholder image so we can continue processing
        print("[SAVE FALLBACK] Creating a blank placeholder image")
        image = np.zeros((224, 224, 3), dtype=np.uint8)  # Create a blank black image
        
        # Draw text on the image to indicate error
        cv2.putText(
            image,
            "Invalid Image Data",
            (30, 112),  # Position in center
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),  # Red color
            2,
        )
    
    # Create temporary file with .jpg extension
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".jpg", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Save image to the temporary file
    try:
        success = cv2.imwrite(temp_path, image)
        if not success:
            print(f"[SAVE ERROR] Failed to write image to {temp_path}")
            return None
            
        print("[SAVE 2] Image saved to temporary file: {0}".format(temp_path))
        return temp_path
        
    except Exception as e:
        print(f"[SAVE ERROR] Error saving image: {str(e)}")
        return None

def process_clock_image(image_path):
    """Process a clock image and return the detected time"""
    print("\n[PROCESS 1] ===== PROCESSING CLOCK IMAGE: {0} =====".format(image_path))
    
    # Load and preprocess the image
    print("\n[PROCESS 2] Loading and preprocessing image")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = einops.rearrange(img, "h w c -> c h w")
    img = torch.Tensor(img)
    img = img.float().to(device)
    img = torch.unsqueeze(img, 0)
    print("[PROCESS 3] Image preprocessed: tensor shape={0}".format(img.shape))

    with torch.no_grad():
        # STN Model Prediction
        print("\n[MODEL 1] Running STN model for spatial transformation")
        pred_st = model_stn(img)
        pred_st = torch.cat([pred_st, torch.ones(1, 1).to(device)], 1)
        Minv_pred = torch.reshape(pred_st, (-1, 3, 3))
        print("[MODEL 2] STN prediction matrix shape: {0}".format(Minv_pred.shape))
        
        # Apply warping
        img_ = warp(img, Minv_pred)
        print("[MODEL 3] Image warped by STN")

        # Main Model Prediction
        print("\n[MODEL 4] Running main model for time prediction")
        pred = model(img_)
        print("[MODEL 5] Raw prediction shape: {0}".format(pred.shape))

        # Get top prediction
        max_pred = torch.argmax(pred, dim=1)
        max_h = max_pred[0] // 60
        max_m = max_pred[0] % 60
        print("[MODEL 6] Raw prediction index: {0}".format(max_pred[0]))
        print("[MODEL 7] Parsed time: {0}h {1}m".format(int(max_h), int(max_m)))

        # Format the time prediction
        time_prediction = f"{int(max_h):02d}:{int(max_m):02d}"
        print("[MODEL 8] Formatted prediction: {0}".format(time_prediction))

    # Create annotated image with prediction
    print("\n[ANNOTATE 1] Creating annotated image with prediction")
    original_img = cv2.imread(image_path)
    
    # Draw predicted time on the image
    cv2.putText(
        original_img,
        f"Detected: {time_prediction}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    
    # Save annotated image
    base_path = Path(image_path)
    annotated_path = base_path.parent / f"annotated_{base_path.name}"
    cv2.imwrite(str(annotated_path), original_img)
    print("[ANNOTATE 2] Annotated image saved to: {0}".format(annotated_path))
    
    result = {
        'detected_time': time_prediction,
        'hours': int(max_h),
        'minutes': int(max_m),
        'original_path': image_path,
        'annotated_path': str(annotated_path)
    }
    
    print("\n[COMPLETE] Clock processing complete: {0}\n".format(result))
    return result

def check_clock_answer(base64_image, expected_answer):
    """
    Check if the clock image shows the expected time
    Returns a tuple: (is_correct, detected_value, annotated_image_path)
    
    A tolerance of ±3 minutes is allowed for correct answers.
    """
    print("\n" + "="*50)
    print("[CHECK 1] CHECKING CLOCK ANSWER (Expected: {0})".format(expected_answer))
    print("="*50)
    
    try:
        # Save the base64 image to a file
        image_path = save_base64_image(base64_image)
        
        if not image_path or not os.path.exists(image_path):
            print("[CHECK ERROR] Failed to save valid image")
            return False, "Error: Invalid image", None
            
        print("[CHECK 2] Base64 image saved to: {0}".format(image_path))
        
        # Process the image
        print("\n[CHECK 3] Processing clock image...")
        result = process_clock_image(image_path)
        
        if not result:
            print("[CHECK ERROR] Failed to process clock image")
            return False, "Error: Processing failed", None
            
        detected_time = result['detected_time']
        annotated_path = result['annotated_path']
        
        # Parse expected answer and format for comparison
        print("\n[PARSE 1] Parsing expected answer: {0}".format(expected_answer))
        expected_minutes_total = 0
        detected_minutes_total = 0
        
        # Parse expected time
        if ":" in expected_answer:
            parts = expected_answer.split(":")
            if len(parts) == 2:
                expected_hour = int(parts[0])
                expected_minute = int(parts[1])
                expected_formatted = f"{expected_hour:02d}:{expected_minute:02d}"
                expected_minutes_total = expected_hour * 60 + expected_minute
                print("[PARSE 2] Parsed expected time: {0}h {1}m ({2} total minutes)".format(
                    expected_hour, expected_minute, expected_minutes_total))
            else:
                expected_formatted = expected_answer
                print("[PARSE 3] Unable to parse expected time properly, using as-is: {0}".format(expected_formatted))
        else:
            # If no colon, try to parse as a number of minutes
            try:
                expected_minutes_total = int(expected_answer)
                expected_hour = expected_minutes_total // 60
                expected_minute = expected_minutes_total % 60
                expected_formatted = f"{expected_hour:02d}:{expected_minute:02d}"
                print("[PARSE 4] Parsed expected minutes: {0} -> {1}h {2}m".format(
                    expected_minutes_total, expected_hour, expected_minute))
            except ValueError:
                expected_formatted = expected_answer
                print("[PARSE 5] Unable to parse expected answer as minutes, using as-is: {0}".format(expected_formatted))
        
        # Parse detected time
        print("\n[PARSE 6] Parsing detected time: {0}".format(detected_time))
        if ":" in detected_time:
            parts = detected_time.split(":")
            if len(parts) == 2:
                detected_hour = int(parts[0])
                detected_minute = int(parts[1])
                detected_minutes_total = detected_hour * 60 + detected_minute
                print("[PARSE 7] Parsed detected time: {0}h {1}m ({2} total minutes)".format(
                    detected_hour, detected_minute, detected_minutes_total))
        
        # Check if the detected time is within ±3 minutes of the expected time
        # First, check if we have valid times to compare
        print("\n[COMPARE 1] Checking if times are within tolerance (±3 minutes)")
        if expected_minutes_total > 0 and detected_minutes_total > 0:
            # Calculate the difference in minutes
            time_difference = abs(detected_minutes_total - expected_minutes_total)
            print("[COMPARE 2] Initial time difference: {0} minutes".format(time_difference))
            
            # Handle edge cases around 12 hours / 24 hours
            if time_difference > 12 * 60 - 3:  # If we're near the day boundary
                time_difference = 24 * 60 - time_difference
                print("[COMPARE 3] Adjusted time difference (day boundary): {0} minutes".format(time_difference))
            
            # Check if within tolerance of 3 minutes
            is_correct = (time_difference <= 3)
            print("[COMPARE 4] Within tolerance? {0} (±3 minutes allowed)".format(is_correct))
            
            # Add tolerance information to the annotated image
            if os.path.exists(annotated_path):
                original_img = cv2.imread(annotated_path)
                if time_difference <= 3:
                    status_text = f"Within tolerance: {time_difference} min diff (±3 min allowed)"
                    color = (0, 255, 0)  # Green for correct
                else:
                    status_text = f"Outside tolerance: {time_difference} min diff (±3 min allowed)"
                    color = (0, 0, 255)  # Red for incorrect
                
                cv2.putText(
                    original_img,
                    status_text,
                    (10, 60),  # Position below the detected time
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                cv2.imwrite(str(annotated_path), original_img)
                print("[COMPARE 5] Updated annotated image with tolerance information")
        else:
            # If we couldn't parse the times properly, fall back to exact string comparison
            is_correct = (detected_time == expected_formatted)
            print("[COMPARE 6] Falling back to exact comparison: {0} == {1} -> {2}".format(
                detected_time, expected_formatted, is_correct))
        
        print("\n[RESULT] Final result: {0}, {1}, {2}".format(is_correct, detected_time, annotated_path))
        print("="*50 + "\n")
        return is_correct, detected_time, annotated_path
    
    except Exception as e:
        # Log the error and return False
        print("\n[ERROR] Error checking clock answer: {0}".format(str(e)))
        print("="*50 + "\n")
        return False, f"Error: {str(e)}", None

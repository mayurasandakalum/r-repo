import os
import pickle
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd

# ------------------------------------------------
# 1) Define function to initialize the same model
#    architecture used during training
# ------------------------------------------------
def get_object_detection_model(num_classes):
    """
    Loads a FasterRCNN_ResNet50_FPN_V2 model and replaces the final predictor
    with one compatible with your custom num_classes.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Replace the pretrained head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ------------------------------------------------
# 2) Inference function
# ------------------------------------------------
def infer(
    model_path,
    image_path,
    class_mapping_path="class_mapping.pkl",
    threshold=0.5,
    proximity_threshold=5
):
    """
    Loads a trained Faster R-CNN model, performs inference on a single image,
    applies a confidence threshold, and then handles overlapping bounding boxes.

    Args:
        model_path (str): Path to the trained .pth model file.
        image_path (str): Path to the input image.
        class_mapping_path (str): Path to the pickled dictionary of
                                  {class_idx: class_name} from training.
        threshold (float): Confidence threshold for filtering predictions.
        proximity_threshold (int): Overlapping bounding box threshold (pixels).
    Returns:
        final_predictions_in_order (list): A list of predicted class names in
                                           left-to-right order.
    """

    # --------------------------
    # Step A: Load model
    # --------------------------
    # First, load the saved class mapping if available
    # This mapping is typically {1: "A", 2: "B", ...} for your custom classes
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, "rb") as f:
            class_mapping = pickle.load(f)
    else:
        # Fallback if you don't have a pickle file
        # or prefer to handle it differently
        class_mapping = {}
        print("WARNING: No class_mapping.pkl found. Labels will be shown as numeric IDs.")

    # Determine how many classes (including background)
    if len(class_mapping) > 0:
        num_classes = max(class_mapping.keys()) + 1  # +1 if background=0 isn't in the mapping
    else:
        # Hardcode if needed
        num_classes = 275  # Example only

    # Create the model structure and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_object_detection_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # --------------------------
    # Step B: Preprocess image
    # --------------------------
    # 1. Open and convert your original image
    image = Image.open(image_path).convert("RGB")

    # 2. Get its dimensions
    width, height = image.size

    # 3. Define fixed final canvas size (512x512)
    final_size = 512

    # 4. Create new 512x512 white background
    extended_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # 5. Compute offsets to center the original image
    offset_x = (final_size - width) // 2
    offset_y = (final_size - height) // 2

    # 6. Paste original image onto the center of the 512x512 canvas
    extended_image.paste(image, (offset_x, offset_y))

    # 7. (Optional) Assign or save as needed
    #extended_image.save("centered_512.png")

    # Now 'image' holds the extended image
    image = extended_image
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    transform = weights.transforms()
    img_tensor = transform(image).unsqueeze(0).to(device)

    # --------------------------
    # Step C: Run inference
    # --------------------------
    with torch.no_grad():
        predictions = model(img_tensor)

    # The predictions will be a list (for each input image) of dicts:
    # predictions[0]["boxes"] -> bounding boxes
    # predictions[0]["labels"] -> predicted class indices
    # predictions[0]["scores"] -> scores for each prediction
    boxes = predictions[0]["boxes"].cpu().numpy()
    labels = predictions[0]["labels"].cpu().numpy()
    scores = predictions[0]["scores"].cpu().numpy()

    # --------------------------
    # Step D: Filter by threshold
    # --------------------------
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []
    for i, score in enumerate(scores):
        if score > threshold:
            filtered_boxes.append(boxes[i])
            filtered_labels.append(labels[i])
            filtered_scores.append(score)

    # --------------------------
    # Step E: Handle overlaps by comparing x1/x2
    # --------------------------
    final_boxes = []
    final_labels = []
    final_scores = []
    visited = [False] * len(filtered_boxes)

    for i in range(len(filtered_boxes)):
        if visited[i]:
            continue
        current_box = filtered_boxes[i]
        x1_current, _, x2_current, _ = current_box

        # Group all boxes that are "close" in x1,x2 coordinates
        overlapping_indices = []
        for j in range(len(filtered_boxes)):
            if not visited[j]:
                x1_other, _, x2_other, _ = filtered_boxes[j]
                if (abs(x1_current - x1_other) <= proximity_threshold or
                        abs(x2_current - x2_other) <= proximity_threshold):
                    overlapping_indices.append(j)

        # Among all overlapping boxes, pick the one with the highest confidence
        best_index = max(overlapping_indices, key=lambda idx: filtered_scores[idx])
        final_boxes.append(filtered_boxes[best_index])
        final_labels.append(filtered_labels[best_index])
        final_scores.append(filtered_scores[best_index])

        # Mark them visited
        for idx in overlapping_indices:
            visited[idx] = True

    # --------------------------
    # Step F: Sort results left->right
    # --------------------------
    # Sort final boxes by x1 coordinate
    sorted_indices = sorted(range(len(final_boxes)), key=lambda i: final_boxes[i][0])

    # --------------------------
    # Step G: Draw and display
    # --------------------------
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    font = ImageFont.load_default()

    final_predictions_in_order = []
    for idx in sorted_indices:
        box = final_boxes[idx]
        label = final_labels[idx]
        score = final_scores[idx]
        x1, y1, x2, y2 = box.astype(int)

        # If you have a mapping from label -> string
        class_name = class_mapping[label] if label in class_mapping else f"Label_{label}"

        # Draw bounding box
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
        text = f"{class_name} : {score:.2f}"
        draw.text((x1, y1 - 10), text, fill="red", font=font)

        final_predictions_in_order.append(class_name)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_draw)
    plt.axis("off")
    plt.show()

    print("Final predicted letters in order: ", final_predictions_in_order)
    return final_predictions_in_order



def filter_bboxes_by_xoverlap(bboxes, overlap_threshold=0.5):
    """
    Removes bounding boxes if their x-ranges overlap with
    any already-kept bounding box more than a certain fraction.
    
    Args:
        bboxes (list): A list of (x, y, w, h) bounding boxes.
        overlap_threshold (float): Fraction of overlap required 
            before we say "they overlap" and remove the new box.
            e.g., 0.5 means if more than half of the box's width 
            overlaps in x, we remove it.
    Returns:
        filtered (list): Filtered bounding boxes (x, y, w, h).
    """
    filtered = []

    for (x, y, w, h) in bboxes:
        x_end = x + w
        # By default, assume no big overlap
        should_skip = False

        for (fx, fy, fw, fh) in filtered:
            fx_end = fx + fw

            # Compute the overlap in x-axis
            overlap_width = min(x_end, fx_end) - max(x, fx)
            if overlap_width > 0:
                # There's some overlap in x dimension.
                # Check fraction of the smaller box's width that is overlapped
                # to decide if it's "significant".
                smaller_width = min(w, fw)
                overlap_ratio = overlap_width / smaller_width

                if overlap_ratio >= overlap_threshold:
                    # Overlaps too much with an existing box
                    should_skip = True
                    break

        if not should_skip:
            filtered.append((x, y, w, h))

    return filtered
# ------------------------------------------------
# EXTRA FUNCTION:
#   Extract sub-images by finding contours 
#   so that each sub-image ideally contains one letter
# ------------------------------------------------
def extract_subimages_with_contours(
    image_path,
    output_dir="temp_subimages",
    bin_threshold=127,
    pad=4
):
    """
    Reads an image (which may contain multiple letters),
    uses OpenCV to find and sort letter contours, 
    and saves individual letter regions as separate images.

    Handles small dots (like the dot above 'i') by using a 
    morphological close operation to merge close contours.

    Args:
        image_path (str): Path to input image
        output_dir (str): Where to save the individual sub-images
        bin_threshold (int): Threshold for binarizing the image

    Returns:
        subimage_paths (list): A list of file paths to the extracted sub-images 
                               (sorted left-to-right).
    """
    # 1. Read image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 3. Threshold -> letters in white on black (inverse or not, depending on your case)
    #    If your letters are black on white, you may prefer THRESH_BINARY_INV
    #    If they are white on black, use THRESH_BINARY
    _, thresh = cv2.threshold(gray, bin_threshold, 255, cv2.THRESH_BINARY_INV)

    # 4. Morphological close to merge "dots" on letters like 'i'
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Get bounding boxes
    bboxes = [cv2.boundingRect(cnt) for cnt in contours]  # (x, y, w, h)

    # 7. Sort bounding boxes left->right (by x)
    bboxes = sorted(bboxes, key=lambda b: b[0])
    
    bboxes = filter_bboxes_by_xoverlap(bboxes, overlap_threshold=0.5)

    # 8. Create output folder if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subimage_paths = []
    img_h, img_w = original.shape[:2]

    for i, (x, y, w, h) in enumerate(bboxes):
        # Keep entire height, just crop on X dimension
        x_start = max(0, x - pad)              # 4 px padding on the left
        x_end = min(img_w, x + w + pad)        # 4 px padding on the right

        # Crop the region horizontally across entire height
        letter_roi = original[:, x_start:x_end]

        sub_name = f"sub_letter_{i}.png"
        sub_path = os.path.join(output_dir, sub_name)
        cv2.imwrite(sub_path, letter_roi)
        subimage_paths.append(sub_path)

    return subimage_paths


# ------------------------------------------------
# 3) Example usage (if run as script)
# ------------------------------------------------
if __name__ == "__main__":
    # Original example code (unchanged):
    # -----------------------------------
    # This part simply shows how you'd call 'infer' on a single image.
    id=45
    image_path = "static/write_img/"+str(id)+".png"
    model_path = "write_model/multi_letter_detector2.pth"
    class_mapping_path = "write_model/class_mapping.pkl"
    class_mapping_path2 = "write_model/class_mapping2.pkl"

    # results = infer(
    #     model_path=model_path,
    #     image_path=image_path,
    #     class_mapping_path=class_mapping_path,
    #     threshold=0.5,
    #     proximity_threshold=5
    # )

    # if os.path.exists(class_mapping_path2):
    #     with open(class_mapping_path2, "rb") as f:
    #         class_mapping2 = pickle.load(f)
    #     # Convert each predicted label from, e.g., "1" -> "A"
    #     results = [class_mapping2[int(i)] for i in results]

    # print("\nPredicted sequence (entire image as input):", "".join(results))

    # -----------------------------------
    # NEW: Split the single image into multiple sub-images
    #      and run the same 'infer' code on each sub-image.
    # -----------------------------------
    print("\nNow splitting the same image into sub-images (one letter each), "
          "then running inference on each sub-image:")

    sub_images = extract_subimages_with_contours(image_path, output_dir=str(id), bin_threshold=127,pad=10)

    final_word = []
    for sub_img_path in sub_images:
        # Use the same inference function on each sub-image
        sub_result = infer(
            model_path=model_path,
            image_path=sub_img_path,
            class_mapping_path=class_mapping_path,
            threshold=0.5,
            proximity_threshold=5
        )
        # Optionally map IDs to actual characters if you have class_mapping2
        if os.path.exists(class_mapping_path2):
            with open(class_mapping_path2, "rb") as f:
                class_mapping2 = pickle.load(f)
            # Convert "sub_result" from numeric classes to letters, then extend final_word
            sub_result = [class_mapping2[int(i)] for i in sub_result]

        # Append recognized letters from this sub-image to the final word
        final_word.extend(sub_result)

    # Print the final combined word after processing sub-images
    print("\nFinal predicted word from sub-images:", "".join(final_word))

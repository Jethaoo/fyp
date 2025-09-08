## YOLO Notebook: Seven-Segment Detection

### Overview
This notebook (`yolo.ipynb`) explores several computer-vision approaches to detect digits on seven-segment LED displays from images. It includes:
- Preprocessing and segmentation to isolate blue LED regions and crop individual digits.
- Multiple contour-based pipelines tuned for seven-segment digits.
- An alternative connected-components approach for comparison.

### Key Files
- `yolo.ipynb`: Main notebook with detection pipelines and visualizations.
- `digits_crops/`: Folder where cropped digit images are saved (e.g., `digit_1.png`).
- `digit.jpg`: Example source image used in some cells (replace with your own).

### Requirements
- Python 3.8+
- Packages:
  - OpenCV (`opencv-python`)
  - NumPy (`numpy`)
  - Matplotlib (`matplotlib`)
  - imutils (`imutils`)

Install with:
```bash
pip install opencv-python numpy matplotlib imutils
```

### What the Notebook Does
1. Detects blue LED regions to locate the display area and derive per-digit spans.
2. Saves per-digit crops to `digits_crops/`.
3. Runs tuned pipelines for seven-segment digits, including morphological closing to merge segments and contour filtering to extract digit-like blobs.
4. Visualizes intermediate steps (grayscale, threshold, morphology) and final bounding boxes.

### Main Functions
- `detect_digits(image_path, ...)`: Generic contour-based detection with configurable thresholds and geometry filters.
- `detect_seven_segment_digits(image_path, debug=False)`: Seven-segment–oriented pipeline. Returns `(digitCnts, output, processed)`:
  - `digitCnts`: list of contours for detected digits (left-to-right).
  - `output`: color image with bounding boxes drawn.
  - `processed`: binary/morphed image used for contouring.
  - `debug`: when `True`, prints additional diagnostics (you can add more conditional logs/plots in the function body).
- `alternative_approach(image_path)`: Connected-components method returning a list of `(x, y, w, h)` regions.

### Typical Workflow
1. Place an input image (e.g., `digit.jpg`) in the project root or use your own path.
2. Run the preprocessing cell to create `digits_crops/` and save digit crops.
3. Call the seven-segment detector on a crop or the full image, e.g.:
```python
image_path = "digits_crops/digit_1.png"
digitCnts, output, processed = detect_seven_segment_digits(image_path, debug=True)
```
4. Inspect detections in the generated plots and adjust parameters if needed.

### Important Parameters and Tuning
- Morphology kernel for closing (merging segments):
  ```python
  kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 12))  # (width, height)
  ```
  - Increase kernel size to bridge larger gaps or merge broken segments.
  - Decrease to avoid merging adjacent digits.
- Contour filtering heuristics (min width/height, area, aspect ratio) are tuned for tall, narrow seven-segment digits; adjust if your display differs.
- Blue mask thresholds (HSV ranges) determine the initial display ROI; widen/narrow for different LED colors or lighting.

### Inputs and Outputs
- Input: path to an image (full display or a cropped digit).
- Outputs:
  - On-disk digit crops in `digits_crops/` (e.g., `digit_1.png`).
  - Visualizations shown inline (original, thresholded, morphed, detections).
  - Function returns (contours, annotated image, processed mask) for further use.

### Troubleshooting
- No digits detected:
  - Try increasing `kernel_close` size (e.g., `(10, 14)`, `(12, 16)`).
  - Relax contour filters (lower area threshold, widen aspect ratio range).
  - Check that thresholding produces white digits on a dark background; invert if needed.
- Too many false positives:
  - Reduce kernel size or add an opening step to remove noise.
  - Tighten area/aspect ratio thresholds.
- Blue ROI not found:
  - Expand HSV ranges or skip ROI masking and run the detector on the full image.

### Notes
- The notebook contains multiple versions of detection functions for experimentation; execute the cell with the function you intend to use last to ensure it is the active definition.
- The `debug` flag is supported in `detect_seven_segment_digits(image_path, debug=False)` to facilitate conditional diagnostics; extend it as needed to toggle extra plots or prints.

### Next Steps
- Add model-based recognition (e.g., a lightweight CNN or template matching) to classify each detected digit.
- Batch process folders of images and export results to CSV/JSON.
- Parameter auto-tuning for different displays and lighting conditions.

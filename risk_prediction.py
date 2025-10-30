# ===============================
# AI-Powered Risk Prediction System
# Fire + Holes + Obstacles + Rain/Water
# ===============================

from ultralytics import YOLO
import cv2
import numpy as np
import time

# -------------------------------
# 1. Load YOLOv8 model
# -------------------------------
model = YOLO("yolov8n.pt")  # YOLOv8 nano for fast inference

# -------------------------------
# 2. Open webcam
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()

# -------------------------------
# 3. Helper functions
# -------------------------------
def estimate_distance(bbox, frame_width):
    """Approximate distance using bounding box width"""
    bbox_width = bbox[2] - bbox[0]
    if bbox_width <= 0:
        return np.inf
    focal_length_px = 800
    real_object_width_m = 0.5  # average human/obstacle width
    distance = (real_object_width_m * focal_length_px) / bbox_width
    return round(distance, 2)

def compute_risk(distance, rel_speed, hazard_type='normal'):
    """Compute risk from distance, relative speed, and hazard type"""
    # Base factors
    d_factor = max(0, (1 - distance / 2.0))       # closer objects increase risk
    v_factor = max(0, rel_speed / 1.0)            # faster approaching increases risk

    # Hazard-specific adjustments
    if hazard_type == 'fire':
        hazard_factor = 1.0
    elif hazard_type == 'hole':
        hazard_factor = 0.8
    elif hazard_type in ['rain', 'water']:
        hazard_factor = 0.6  # slippery conditions
    else:
        hazard_factor = 0.1  # normal obstacles

    risk = 0.6 * d_factor + 0.3 * v_factor + 0.1 * hazard_factor
    return min(risk, 1.0)

# Track previous distances for speed estimation
prev_distances = {}
prev_times = {}

print("✅ Starting AI-Powered Risk Prediction...")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]

    risk_score = 0.0

    # -------------------------------
    # 4. Detect obstacles using YOLO
    # -------------------------------
    results = model(frame, verbose=False)[0]

    for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = box
        cls = int(results.boxes.cls[i].cpu().numpy())
        label = model.names[cls]
        conf = float(results.boxes.conf[i].cpu().numpy())

        if conf < 0.5:
            continue

        # Estimate distance
        dist = estimate_distance((x1, y1, x2, y2), W)

        # Estimate relative speed
        now = time.time()
        prev_dist = prev_distances.get(label, None)
        prev_t = prev_times.get(label, now)
        rel_speed = 0
        if prev_dist is not None:
            dt = now - prev_t
            rel_speed = (prev_dist - dist) / dt if dt > 0 else 0
        prev_distances[label] = dist
        prev_times[label] = now

        # Compute risk for obstacle
        r = compute_risk(dist, rel_speed)
        risk_score = max(risk_score, r)

        # Draw detection
        color = (0, 0, 255) if r > 0.7 else (0, 255, 255) if r > 0.4 else (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Dist: {dist:.2f}m Risk: {r:.2f}",
                    (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # -------------------------------
    # 5. Detect fire using color thresholds
    # -------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([0, 150, 150])
    upper_fire = np.array([35, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
    fire_pixels = cv2.countNonZero(fire_mask)
    fire_detected = fire_pixels > 500  # adjust threshold

    if fire_detected:
        r_fire = 1.0
        risk_score = max(risk_score, r_fire)
        cv2.putText(frame, "🔥 FIRE DETECTED!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # -------------------------------
    # 6. Detect holes (roughly) using contours
    # -------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 2)
            hole_detected = True
    if hole_detected:
        r_hole = 0.8
        risk_score = max(risk_score, r_hole)
        cv2.putText(frame, "⚠️ HOLE DETECTED!", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)

    # -------------------------------
    # 7. Detect rain / water puddles
    # -------------------------------
    # Rain: bright reflections + motion in frame (simplified)
    gray_blur = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0)
    _, thresh = cv2.threshold(gray_blur, 200, 255, cv2.THRESH_BINARY)
    rain_pixels = cv2.countNonZero(thresh)
    rain_detected = rain_pixels > 3000  # adjust threshold

    # Water puddles: detect blueish / reflective areas
    lower_water = np.array([90, 50, 50])
    upper_water = np.array([140, 255, 255])
    water_mask = cv2.inRange(hsv, lower_water, upper_water)
    water_pixels = cv2.countNonZero(water_mask)
    water_detected = water_pixels > 2000

    if rain_detected:
        r_rain = 0.6
        risk_score = max(risk_score, r_rain)
        cv2.putText(frame, "🌧️ RAIN DETECTED!", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)

    if water_detected:
        r_water = 0.6
        risk_score = max(risk_score, r_water)
        cv2.putText(frame, "💧 WATER DETECTED!", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

    # -------------------------------
    # 8. Display overall risk level
    # -------------------------------
    if risk_score >= 0.7:
        status = f"🚨 HIGH RISK ({risk_score:.2f})"
        color = (0, 0, 255)
    elif risk_score >= 0.4:
        status = f"⚠️  MEDIUM RISK ({risk_score:.2f})"
        color = (0, 255, 255)
    else:
        status = f"✅ SAFE ({risk_score:.2f})"
        color = (0, 255, 0)

    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # -------------------------------
    # 9. Show window
    # -------------------------------
    cv2.imshow("AI Risk Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

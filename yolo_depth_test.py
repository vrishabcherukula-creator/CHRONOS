import cv2
import torch
from ultralytics import YOLO

# -----------------------------
# LOAD MODELS
# -----------------------------

# YOLO object detection
yolo = YOLO("yolo11n.pt")

# MiDaS depth estimation
midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)

midas.eval()

# MiDaS transformation
midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.small_transform


# -----------------------------
# OPEN WEBCAM
# -----------------------------

camera = cv2.VideoCapture(0)


while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not access camera")
        break


    # -----------------------------
    # YOLO DETECTION
    # -----------------------------

    results = yolo(frame, conf=0.4)

    output = frame.copy()


    # -----------------------------
    # MiDaS DEPTH
    # -----------------------------

    img = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    input_batch = transform(img)

    with torch.no_grad():
        prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

    depth_map = prediction.cpu().numpy()


    # -----------------------------
    # PROCESS YOLO OBJECTS
    # -----------------------------

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # Object center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)


        # Get depth at object center
        depth_value = depth_map[
            center_y,
            center_x
        ]


        # Draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Draw center
        cv2.circle(
            output,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )


        # Display depth
        cv2.putText(
            output,
            f"Depth: {depth_value:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


    # -----------------------------
    # DISPLAY DEPTH MAP
    # -----------------------------

    depth_display = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = depth_display.astype("uint8")

    depth_display = cv2.cvtColor(
        depth_display,
        cv2.COLOR_GRAY2BGR
    )


    # Resize
    camera_view = cv2.resize(
        output,
        (640, 480)
    )

    depth_view = cv2.resize(
        depth_display,
        (640, 480)
    )


    # Side-by-side view
    combined = cv2.hconcat([
        camera_view,
        depth_view
    ])


    cv2.imshow(
        "CHRONOS - YOLO + Depth",
        combined
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
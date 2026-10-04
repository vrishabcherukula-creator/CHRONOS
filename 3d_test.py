import cv2
import torch
from ultralytics import YOLO

# --------------------------------
# LOAD MODELS
# --------------------------------

yolo = YOLO("yolo11n.pt")

midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)

midas.eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

depth_transform = midas_transforms.small_transform


# --------------------------------
# OPEN CAMERA
# --------------------------------

camera = cv2.VideoCapture(0)


while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not access camera")
        break

    output = frame.copy()


    # --------------------------------
    # DEPTH ESTIMATION
    # --------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    depth_input = depth_transform(rgb_frame)

    with torch.no_grad():
        depth_prediction = midas(depth_input)

    depth_prediction = torch.nn.functional.interpolate(
        depth_prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

    depth_map = depth_prediction.cpu().numpy()


    # --------------------------------
    # YOLO DETECTION
    # --------------------------------

    results = yolo(
        frame,
        conf=0.4
    )


    # --------------------------------
    # PROCESS OBJECTS
    # --------------------------------

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # Object center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)


        # --------------------------------
        # GET DEPTH
        # --------------------------------

        depth_value = depth_map[
            center_y,
            center_x
        ]


        # --------------------------------
        # NORMALIZE X AND Y
        # --------------------------------

        frame_height, frame_width = frame.shape[:2]

        normalized_x = center_x / frame_width

        normalized_y = center_y / frame_height


        # --------------------------------
        # NORMALIZE DEPTH
        # --------------------------------

        depth_min = depth_map.min()
        depth_max = depth_map.max()

        normalized_z = (
            (depth_value - depth_min)
            / (depth_max - depth_min + 1e-6)
        )


        # --------------------------------
        # DRAW OBJECT
        # --------------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            output,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )


        # --------------------------------
        # DISPLAY 3D COORDINATES
        # --------------------------------

        cv2.putText(
            output,
            f"X: {normalized_x:.2f}",
            (x1, y1 - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Y: {normalized_y:.2f}",
            (x1, y1 - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Z: {normalized_z:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )


    # --------------------------------
    # SHOW OUTPUT
    # --------------------------------

    cv2.imshow(
        "CHRONOS - 3D Coordinates",
        output
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
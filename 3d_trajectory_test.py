import cv2
import torch
import math
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


# --------------------------------
# STORE 3D TRAJECTORIES
# --------------------------------

spatial_tracks = {}


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
    # NORMALIZE DEPTH
    # --------------------------------

    depth_min = depth_map.min()
    depth_max = depth_map.max()


    # --------------------------------
    # YOLO DETECTION
    # --------------------------------

    results = yolo.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5
    )


    # --------------------------------
    # PROCESS OBJECTS
    # --------------------------------

    if results[0].boxes.id is not None:

        ids = results[0].boxes.id.cpu().numpy()
        
        for box, object_id in zip(
            results[0].boxes,
            ids
        ):

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            # Object center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # --------------------------------
            # 3D COORDINATES
            # --------------------------------

            normalized_x = center_x / frame.shape[1]

            normalized_y = center_y / frame.shape[0]

            depth_value = depth_map[
                center_y,
                center_x
            ]

            normalized_z = (
                (depth_value - depth_min)
                / (depth_max - depth_min + 1e-6)
            )

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # Object center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # --------------------------------
        # 3D COORDINATES
        # --------------------------------

        normalized_x = center_x / frame.shape[1]

        normalized_y = center_y / frame.shape[0]

        depth_value = depth_map[
            center_y,
            center_x
        ]

        normalized_z = (
            (depth_value - depth_min)
            / (depth_max - depth_min + 1e-6)
        )


        # --------------------------------
        # CREATE TRACK
        # --------------------------------

        if object_id not in spatial_tracks:

            spatial_tracks[object_id] = []


        # Store 3D position
        spatial_tracks[object_id].append(
            (
                normalized_x,
                normalized_y,
                normalized_z
            )
        )


        # Keep last 30 points
        if len(spatial_tracks[object_id]) > 30:

            spatial_tracks[object_id].pop(0)


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
        # DRAW 3D TRAJECTORY
        # --------------------------------

        points = spatial_tracks[object_id]

        for i in range(1, len(points)):

            previous_point = points[i - 1]
            current_point = points[i]

            previous_x = int(
                previous_point[0] * frame.shape[1]
            )

            previous_y = int(
                previous_point[1] * frame.shape[0]
            )

            current_x = int(
                current_point[0] * frame.shape[1]
            )

            current_y = int(
                current_point[1] * frame.shape[0]
            )

            cv2.line(
                output,
                (previous_x, previous_y),
                (current_x, current_y),
                (0, 255, 0),
                2
            )


        # --------------------------------
        # DISPLAY 3D POSITION
        # --------------------------------

        cv2.putText(
            output,
            f"X: {normalized_x:.2f}",
            (x1, y1 - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Y: {normalized_y:.2f}",
            (x1, y1 - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Z: {normalized_z:.2f}",
            (x1, y1 - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )


    # --------------------------------
    # SHOW
    # --------------------------------

    cv2.imshow(
        "CHRONOS - 3D Trajectory",
        output
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
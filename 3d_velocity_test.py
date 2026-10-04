import cv2
import torch
import math
from ultralytics import YOLO


# -----------------------------
# LOAD MODELS
# -----------------------------

yolo = YOLO("yolo11n.pt")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

depth_transform = midas_transforms.small_transform


# -----------------------------
# CAMERA
# -----------------------------

camera = cv2.VideoCapture(0)


# -----------------------------
# 3D TRACK STORAGE
# -----------------------------

spatial_tracks = {}


# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        break


    # -----------------------------
    # DEPTH ESTIMATION
    # -----------------------------

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


    # -----------------------------
    # YOLO TRACKING
    # -----------------------------

    results = yolo.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5
    )

    output = frame.copy()


    if results[0].boxes.id is not None:

        boxes = results[0].boxes

        ids = boxes.id.cpu().numpy()


        for box, object_id in zip(boxes, ids):

            object_id = int(object_id)


            # -----------------------------
            # BOUNDING BOX
            # -----------------------------

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            # -----------------------------
            # OBJECT CENTER
            # -----------------------------

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # -----------------------------
            # 3D COORDINATES
            # -----------------------------

            frame_height, frame_width = frame.shape[:2]

            x_3d = center_x / frame_width
            y_3d = center_y / frame_height


            depth_value = depth_map[
                center_y,
                center_x
            ]


            depth_min = depth_map.min()
            depth_max = depth_map.max()

            z_3d = (
                (depth_value - depth_min)
                /
                (depth_max - depth_min + 1e-6)
            )


            # -----------------------------
            # STORE 3D POSITION
            # -----------------------------

            if object_id not in spatial_tracks:

                spatial_tracks[object_id] = []


            spatial_tracks[object_id].append(
                (
                    x_3d,
                    y_3d,
                    z_3d
                )
            )


            # Keep last 30 positions

            if len(spatial_tracks[object_id]) > 30:

                spatial_tracks[object_id].pop(0)


            # -----------------------------
            # DEFAULT VALUES
            # -----------------------------

            vx = 0
            vy = 0
            vz = 0

            speed_3d = 0


            # -----------------------------
            # 3D VELOCITY
            # -----------------------------

            points = spatial_tracks[object_id]


            if len(points) >= 5:

                previous_point = points[-5]

                current_point = points[-1]


                dx = (
                    current_point[0]
                    -
                    previous_point[0]
                )

                dy = (
                    current_point[1]
                    -
                    previous_point[1]
                )

                dz = (
                    current_point[2]
                    -
                    previous_point[2]
                )


                # Velocity components

                vx = dx / 5

                vy = dy / 5

                vz = dz / 5


                # 3D velocity magnitude

                speed_3d = math.sqrt(
                    vx ** 2
                    +
                    vy ** 2
                    +
                    vz ** 2
                )


            # -----------------------------
            # DRAW OBJECT
            # -----------------------------

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


            # -----------------------------
            # DISPLAY INFORMATION
            # -----------------------------

            cv2.putText(
                output,
                f"ID: {object_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


            cv2.putText(
                output,
                f"X: {x_3d:.2f}  Y: {y_3d:.2f}  Z: {z_3d:.2f}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                2
            )


            cv2.putText(
                output,
                f"3D Velocity: {speed_3d:.4f}",
                (x1, y2 + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )


    # -----------------------------
    # SHOW FRAME
    # -----------------------------

    cv2.imshow(
        "CHRONOS - 3D Velocity",
        output
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()

cv2.destroyAllWindows()
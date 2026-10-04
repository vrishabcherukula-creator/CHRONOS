import cv2
import torch
import math
from ultralytics import YOLO


# -----------------------------
# LOAD MODELS
# -----------------------------

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


# -----------------------------
# CAMERA
# -----------------------------

camera = cv2.VideoCapture(0)


# -----------------------------
# TRACK STORAGE
# -----------------------------

spatial_tracks = {}

direction_history = {}


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
            # CENTER
            # -----------------------------

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # -----------------------------
            # NORMALIZED 3D POSITION
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
            # CREATE TRACK
            # -----------------------------

            if object_id not in spatial_tracks:

                spatial_tracks[object_id] = []

                direction_history[object_id] = []


            spatial_tracks[object_id].append(
                (
                    x_3d,
                    y_3d,
                    z_3d
                )
            )


            if len(spatial_tracks[object_id]) > 30:

                spatial_tracks[object_id].pop(0)


            # -----------------------------
            # DEFAULT VALUES
            # -----------------------------

            direction = "STATIONARY"

            turn_status = "NO TURN"


            # -----------------------------
            # CALCULATE MOTION
            # -----------------------------

            points = spatial_tracks[object_id]


            if len(points) >= 5:

                previous = points[-5]

                current = points[-1]


                dx = current[0] - previous[0]

                dy = current[1] - previous[1]

                dz = current[2] - previous[2]


                vx = dx / 5

                vy = dy / 5

                vz = dz / 5


                threshold = 0.005


                # -----------------------------
                # DIRECTION
                # -----------------------------

                if (
                    abs(vx) < threshold
                    and
                    abs(vy) < threshold
                    and
                    abs(vz) < threshold
                ):

                    direction = "STATIONARY"


                elif (
                    abs(vx) >= abs(vy)
                    and
                    abs(vx) >= abs(vz)
                ):

                    if vx > threshold:

                        direction = "RIGHT"

                    elif vx < -threshold:

                        direction = "LEFT"


                elif (
                    abs(vy) >= abs(vx)
                    and
                    abs(vy) >= abs(vz)
                ):

                    if vy > threshold:

                        direction = "DOWN"

                    elif vy < -threshold:

                        direction = "UP"


                else:

                    direction = "DEPTH CHANGE"


                # -----------------------------
                # SAVE DIRECTION
                # -----------------------------

                direction_history[object_id].append(
                    direction
                )


                if len(direction_history[object_id]) > 8:

                    direction_history[object_id].pop(0)


                # -----------------------------
                # TURN DETECTION
                # -----------------------------

                history = direction_history[object_id]


                if len(history) >= 6:

                    old_direction = history[-6]

                    recent_direction = history[-1]


                    if (
                        old_direction != recent_direction
                        and
                        old_direction != "STATIONARY"
                        and
                        recent_direction != "STATIONARY"
                        and
                        old_direction != "DEPTH CHANGE"
                        and
                        recent_direction != "DEPTH CHANGE"
                    ):

                        turn_status = (
                            f"TURN: "
                            f"{old_direction} -> "
                            f"{recent_direction}"
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
            # DISPLAY ID
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


            # -----------------------------
            # DISPLAY XYZ
            # -----------------------------

            cv2.putText(
                output,
                f"XYZ: "
                f"{x_3d:.2f}, "
                f"{y_3d:.2f}, "
                f"{z_3d:.2f}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                2
            )


            # -----------------------------
            # DISPLAY DIRECTION
            # -----------------------------

            cv2.putText(
                output,
                f"Direction: {direction}",
                (x1, y2 + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                2
            )


            # -----------------------------
            # DISPLAY TURN
            # -----------------------------

            cv2.putText(
                output,
                turn_status,
                (x1, y2 + 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )


    # -----------------------------
    # SHOW
    # -----------------------------

    cv2.imshow(
        "CHRONOS - 3D Turn Detection",
        output
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# -----------------------------
# CLEANUP
# -----------------------------

camera.release()

cv2.destroyAllWindows()
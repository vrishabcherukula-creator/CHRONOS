import cv2
import math
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO


# =========================
# LOAD MODELS
# =========================

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


# =========================
# CAMERA
# =========================

camera = cv2.VideoCapture(0)


# =========================
# 3D TRAJECTORY STORAGE
# =========================

spatial_tracks = {}


# =========================
# 3D PLOT
# =========================

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = camera.read()

    if not ret:
        break

    results = yolo.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5
    )

    # =========================
    # DEPTH ESTIMATION
    # =========================

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

    frame_height, frame_width = frame.shape[:2]

    # =========================
    # DETECTIONS
    # =========================

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, object_id in zip(boxes, ids):

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            object_id = int(object_id)

            # =========================
            # NORMALIZED 3D POSITION
            # =========================

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

            # =========================
            # INITIALIZE TRACK
            # =========================

            if object_id not in spatial_tracks:
                spatial_tracks[object_id] = []

            # =========================
            # STORE 3D POSITION
            # =========================

            spatial_tracks[object_id].append(
                (x_3d, y_3d, z_3d)
            )

            if len(spatial_tracks[object_id]) > 30:
                spatial_tracks[object_id].pop(0)

            # =========================
            # DRAW OBJECT
            # =========================

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {object_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"XYZ: ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})",
                (int(x1), int(y2) + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

    # =========================
    # UPDATE 3D PLOT
    # =========================

    ax.clear()

    for object_id, points in spatial_tracks.items():

        if len(points) < 2:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        ax.plot(
            xs,
            ys,
            zs,
            marker="o",
            label=f"Object {object_id}"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("CHRONOS - 3D Object Trajectory")

    if spatial_tracks:
        ax.legend()

    plt.pause(0.001)

    # =========================
    # DISPLAY CAMERA
    # =========================

    cv2.imshow(
        "CHRONOS - 3D Visualization",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

plt.ioff()
plt.close()
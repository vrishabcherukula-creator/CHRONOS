import cv2
import math
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ORB feature detector
orb = cv2.ORB_create(nfeatures=500)
previous_descriptors = None
previous_keypoints = None
previous_frame = None

# Load YOLO model
model = YOLO("yolo11n.pt")

# Load MiDaS depth model
midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)

midas.eval()

# Load MiDaS transformation
midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

depth_transform = midas_transforms.small_transform

# Open webcam
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# Store positions of each object
tracks = {}
# Store recent velocities of each object
velocity_history = {}
turn_counters = {}
previous_directions = {}
behavior_history = {}
event_history = {}
movement_history = {}
# Store 3D positions of each object
spatial_tracks = {}
# Store sparse 3D feature points
point_cloud = []
# Camera-motion compensated point cloud
compensated_point_cloud = []

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")



while True:

    # Read camera frame
    ret, frame = camera.read()

    if not ret:
        print("Could not access the camera")
        break

    # --------------------------------
    # ORB FEATURE MATCHING
    # --------------------------------

    if (
        previous_descriptors is not None
        and descriptors is not None
        and previous_keypoints is not None
        and previous_frame is not None
    ):

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )

        matches = matcher.match(
            previous_descriptors,
            descriptors
        )

        matches = sorted(
            matches,
            key=lambda match: match.distance
        )

        # Keep only strong ORB matches
        good_matches = [
            match for match in matches[:50]
            if match.distance < 40
        ]

    else:
        good_matches = []

# --------------------------------
# DRAW ORB MATCHES
# --------------------------------

    if good_matches:
        matched_frame = cv2.drawMatches(
            frame,
            keypoints,
            frame,
            keypoints,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        previous_descriptors = descriptors
        previous_keypoints = keypoints
        previous_frame = frame.copy()

        # --------------------------------
        # CAMERA MOTION ESTIMATION
        # --------------------------------

        camera_motion = "UNKNOWN"

        if len(good_matches) >= 8:

            previous_points = []
            current_points = []

            for match in good_matches:
                previous_points.append(
                    previous_keypoints[match.queryIdx].pt
                )
                current_points.append(
                    keypoints[match.trainIdx].pt
                )

            previous_points = __import__("numpy").float32(
                previous_points
            )

            current_points = __import__("numpy").float32(
                current_points
            )

            transform, mask = cv2.estimateAffinePartial2D(
                previous_points,
                current_points
            )

            if transform is not None:

                translation_x = transform[0, 2]
                translation_y = transform[1, 2]

                if abs(translation_x) > 2:
                    camera_motion = "HORIZONTAL"

                elif abs(translation_y) > 2:
                    camera_motion = "VERTICAL"

                else:
                    camera_motion = "STABLE"


    # Detect and track objects
    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5
    )

    # Copy frame for drawing
    output = frame.copy()

    cv2.putText(
        output,
        "CHRONOS | SPATIAL MOTION INTELLIGENCE",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        output,
        "PAST -> PRESENT -> FUTURE",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

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
    # DEPTH MAP VISUALIZATION
    # --------------------------------

    depth_normalized = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = depth_normalized.astype("uint8")

    cv2.imshow(
        "CHRONOS | DEPTH MAP",
        depth_display
    )

    # --------------------------------
    # SPARSE 3D POINT CLOUD
    # --------------------------------

    sparse_3d_points = []

    if len(good_matches) >= 8:

        for match in good_matches:

            x, y = keypoints[match.trainIdx].pt

            x = int(x)
            y = int(y)

            if (
                0 <= x < depth_map.shape[1]
                and 0 <= y < depth_map.shape[0]
            ):

                z = float(depth_map[y, x])

                frame_height, frame_width = frame.shape[:2]

                x_3d = x / frame_width
                y_3d = y / frame_height

                if z > 0:
                    
                    sparse_3d_points.append(
                        (
                            x_3d,
                            y_3d,
                            z
                        )
                    )

    point_cloud.extend(sparse_3d_points)

    if len(point_cloud) > 2000:
        point_cloud = point_cloud[-2000:]

    # Camera-motion compensated point cloud
    compensated_point_cloud = point_cloud.copy()

    # Default dashboard values
    dominant_direction = "Stationary"
    prediction_confidence = 0
    average_speed = 0

    # --------------------------------
    # SCENE UNDERSTANDING
    # --------------------------------

    scene_relationships = [] 


    # Check if tracking IDs exist
    if results[0].boxes.id is not None:

    

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        active_ids = set(ids.astype(int))


        # Process every object
        for box, object_id in zip(boxes, ids):

            x1, y1, x2, y2 = box

            # Calculate center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Get relative depth at object center
            depth_value = depth_map[
                center_y,
                center_x
            ]
            object_id = int(object_id)

            # --------------------------------
            # 3D POSITION
            # --------------------------------

            frame_height, frame_width = frame.shape[:2]
            x_3d = center_x / frame_width
            y_3d = center_y / frame_height

            depth_min = depth_map.min()
            depth_max = depth_map.max()

            z_3d = (
                (depth_value - depth_min)
                /
                (depth_max - depth_min + 1e-6)
            )


            # Create history for new object
            if object_id not in tracks:
                tracks[object_id] = []
                spatial_tracks[object_id] = []
                velocity_history[object_id] = []
                previous_directions[object_id] = "Stationary"
                turn_counters[object_id] = 0
                behavior_history[object_id] = []
                event_history[object_id] = []
                movement_history[object_id] = []

            # Save current position
            tracks[object_id].append(
                (center_x, center_y)
            )


            # Keep only last 30 positions
            if len(tracks[object_id]) > 30:
                tracks[object_id].pop(0)

            # --------------------------------
            # SAVE 3D POSITION
            # --------------------------------
            spatial_tracks[object_id].append(
                (x_3d, y_3d, z_3d)
            )

            if len(spatial_tracks[object_id]) > 30:
                spatial_tracks[object_id].pop(0)

            # Draw ID
            cv2.putText(
                output,
                f"ID: {object_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # Draw actual center
            cv2.circle(
                output,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1
            )


            # Get trajectory
            points = tracks[object_id]


            # --------------------------------
            # DEFAULT VALUES
            # --------------------------------

            movement = 0
            speed = 0
            # 3D velocity
            speed_3d = 0
            vx_3d = 0
            vy_3d = 0
            vz_3d = 0
            direction = "Stationary"
            status = "STATIONARY"
            path_type = "UNKNOWN"
            turn_status = "NO TURN"
            behavior = "UNKNOWN"
            dominant_behavior = "UNKNOWN"
            behavior_confidence = 0
            event = ""
            temporal_state = "STABLE"

            predicted_x = center_x
            predicted_y = center_y


            # --------------------------------
            # MOVEMENT + VELOCITY
            # --------------------------------

            if len(points) >= 5:

                # Position from 5 frames ago
                previous_x, previous_y = points[-5]

                # Current position
                current_x, current_y = points[-1]


                # Difference in position
                dx = current_x - previous_x
                dy = current_y - previous_y

                # Determine basic path type
                if abs(dx) > abs(dy):
                    path_type = "Horizontal"
                else:
                    path_type = "Vertical"


                # Calculate movement
                movement = math.sqrt(
                    dx**2 + dy**2
                )


                # Calculate velocity
                vx = dx / 5
                vy = dy / 5



                # --------------------------------
                # 3D VELOCITY
                # --------------------------------

                spatial_points = spatial_tracks[object_id]

                if len(spatial_points) >= 5:
                    previous_3d = spatial_points[-5]
                    current_3d = spatial_points[-1]

                    dx_3d = current_3d[0] - previous_3d[0]
                    dy_3d = current_3d[1] - previous_3d[1]

                    dz_3d = current_3d[2] - previous_3d[2]

                    vx_3d = dx_3d / 5

                    vy_3d = dy_3d / 5

                    vz_3d = dz_3d / 5

                    speed_3d = math.sqrt(
                    vx_3d ** 2
                    +
                    vy_3d ** 2
                    +
                    vz_3d ** 2
                    )


                # Store recent velocity
                velocity_history[object_id].append(
                    (vx, vy)
                )

                # Keep only last 5 velocities
                if len(velocity_history[object_id]) > 5:
                    velocity_history[object_id].pop(0)

                # Calculate average velocity
                avg_vx = sum(
                    v[0] for v in velocity_history[object_id]
                ) / len(velocity_history[object_id])
                avg_vy = sum(
                    v[1] for v in velocity_history[object_id]
                ) / len(velocity_history[object_id])

                # Calculate smoothed speed
                speed = math.sqrt(
                    avg_vx**2 + avg_vy**2
                )

                # --------------------------------
                # MOVING / STATIONARY
                # --------------------------------

                if movement > 5:

                    status = "MOVING"

                else:
                    status = "STATIONARY"

                # Detect movement start
                if object_id not in previous_directions:
                    previous_directions[object_id] = "STATIONARY"

                previous_status = previous_directions[object_id]

                if previous_status == "STATIONARY" and status == "MOVING":
                    event = "STARTED MOVING"
                elif previous_status == "MOVING" and status == "STATIONARY":
                    event = "STOPPED MOVING"
                else:
                    event = ""

                if event != "":
                    event_history[object_id].append(event)

                    # Keep only the latest 5 events
                    if len(event_history[object_id]) > 5:
                        event_history[object_id].pop(0)

                previous_directions[object_id] = status

                # --------------------------------
                # DIRECTION
                # --------------------------------

                if abs(dx) > abs(dy):

                    if dx > 0:
                        direction = "Right"
                    else:
                        direction = "Left"

                else:

                    if dy > 0:
                        direction = "Down"
                    else:
                        direction = "Up"

                if (
                    previous_directions[object_id] != "Stationary"
                    and direction != previous_directions[object_id]
                ):
                    turn_counters[object_id] += 1
                else:
                    turn_counters[object_id] = 0

                if turn_counters[object_id] >= 3:
                    turn_status = "TURN DETECTED"

                    previous_directions[object_id] = direction
                    turn_counters[object_id] = 0
                else:
                    turn_status = "NO TURN"

                if turn_status == "TURN DETECTED":
                    behavior = "TURNING"
                elif status == "MOVING":
                    behavior = "MOVING STRAIGHT"
                else:
                    behavior = "UNKNOWN"

                # Save current behavior
                behavior_history[object_id].append(behavior)

                # Keep only the latest 10 behaviors
                if len(behavior_history[object_id]) > 10:
                    behavior_history[object_id].pop(0)

                # --------------------------------
                # BEHAVIOR SUMMARY
                # --------------------------------

                if len(behavior_history[object_id]) > 0:

                    behavior_counts = {}

                    for state in behavior_history[object_id]:
                        if state not in behavior_counts:
                            behavior_counts[state] = 0
                        behavior_counts[state] += 1

                    dominant_behavior = max(
                        behavior_counts,
                        key=behavior_counts.get
                    )

                    behavior_confidence = (
                        behavior_counts[dominant_behavior]
                        / len(behavior_history[object_id])
                    ) * 100
                else:
                    dominant_behavior = "UNKNOWN"
                    behavior_confidence = 0


                # --------------------------------
                # MOVEMENT HISTORY
                # --------------------------------

                movement_history[object_id].append({
                    "status": status,
                    "direction": direction,
                    "speed": speed
                })

                if len(movement_history[object_id]) > 30:
                    movement_history[object_id].pop(0)

                recent_directions = [
                    item["direction"]
                    for item in movement_history[object_id]
                    if item["direction"] != "Stationary"
                ]

                # TEMPORAL MOTION PROFILE
                recent_speeds = [
                    item["speed"]
                    for item in movement_history[object_id]
                ]

                if recent_speeds:
                    average_speed = sum(recent_speeds) / len(recent_speeds)
                else:
                    average_speed = 0

                # SPEED VARIATION
                if recent_speeds:
                    speed_variation = max(recent_speeds) - min(recent_speeds)
                else:
                    speed_variation = 0

                # TEMPORAL MOTION STATE
                if speed_variation < 2:
                    temporal_state = "STABLE"
                else:
                    temporal_state = "CHANGING"

                if recent_directions:
                    dominant_direction = max(
                    set(recent_directions),
                    key=recent_directions.count
                    )                  
                else:
                    dominant_direction = "Stationary"

                # PREDICTION CONFIDENCE
                if recent_directions:
                    dominant_count = recent_directions.count(dominant_direction)
                    prediction_confidence = (
                        dominant_count / len(recent_directions)
                    ) * 100
                else:
                    prediction_confidence = 0
                    # --------------------------------
                    # PREDICTION
                    # --------------------------------

                    predicted_x = current_x + (avg_vx * 10)
                    predicted_y = current_y + (avg_vy * 10)

                if dominant_direction == direction:
                    predicted_x = current_x + (avg_vx * 12)
                    predicted_y = current_y + (avg_vy * 12)

                future_points = []
                for step in range(1, 6):
                    future_x = current_x + (avg_vx * 12 * step / 5)
                    future_y = current_y + (avg_vy * 12 * step / 5)

                    future_points.append(
                    (int(future_x), int(future_y))
                    )

            # --------------------------------
            # TRAJECTORY DISTANCE
            # --------------------------------

            trajectory_distance = 0

            for i in range(1, len(points)):
                x1_track, y1_track = points[i - 1]
                x2_track, y2_track = points[i]

                distance = math.sqrt(
                    (x2_track - x1_track) ** 2 +
                    (y2_track - y1_track) ** 2
                )

                trajectory_distance += distance



            # --------------------------------
            # DRAW TRAJECTORY
            # --------------------------------

            for i in range(1, len(points)):

                cv2.line(
                    output,
                    points[i - 1],
                    points[i],
                    (0, 255, 0),
                    2
                )


            # --------------------------------
            # DRAW PREDICTED POSITION
            # --------------------------------

            if status == "MOVING":

                cv2.circle(
                    output,
                    (int(predicted_x), int(predicted_y)),
                    8,
                    (255, 0, 255),
                    -1
                )

                if status == "MOVING":
                    cv2.putText(
                        output,
                        "PREDICTED",
                        (int(predicted_x) + 10, int(predicted_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2
                    )

                if status == "MOVING" and len(future_points) > 1:
                    for i in range(1, len(future_points)):
                        cv2.line(
                            output,
                            future_points[i - 1],
                            future_points[i],
                            (255, 0, 255),
                            2
                        )

                cv2.line(
                    output,
                    (center_x, center_y),
                    (int(predicted_x), int(predicted_y)),
                    (255, 0, 255),
                    2
                )


            # --------------------------------
            # DISPLAY INFORMATION
            # --------------------------------

            cv2.putText(
                output,
                f"Depth: {depth_value:.2f}",
                (int(x1), int(y2) + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output,
                f"{status} | {direction}",
                (int(x1), int(y2) + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output,
                f"XYZ: ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})",
                (int(x1), int(y2) + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"3D Velocity: {speed_3d:.4f}",
                (int(x1), int(y2) + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output,
                f"Movement: {movement:.1f} px",
                (int(x1), int(y2) + 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )



            cv2.putText(
                output,
                f"Velocity: {speed:.1f} px/frame",
                (int(x1), int(y2) + 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )


            cv2.putText(
                output,
                f"Path Type: {path_type}",
                (int(x1), int(y2) + 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            #cv2.putText(
             #   output,
              #  f"Turn Status: {turn_status}",
               # (int(x1), int(y2) + 125),
                #cv2.FONT_HERSHEY_SIMPLEX,
                #0.45,
                #(0, 255, 255),
                #2
            #)

            #cv2.putText(
             #   output,
              #  f"Behavior: {behavior}",
               # (int(x1), int(y2) + 145),
                #cv2.FONT_HERSHEY_SIMPLEX,
                #0.45,
                #(0, 255, 255),
                #2
            #) 

            #cv2.putText(
             #   output,
              #  f"Dominant Behavior: {dominant_behavior}",
               # (int(x1), int(y2) + 165),
                #cv2.FONT_HERSHEY_SIMPLEX,
                #0.45,
                #(0, 255, 255),
                #2
            #)


            cv2.putText(
                output,
                f"Event: {event}",
                (int(x1), int(y2) + 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )


            start_count = event_history[object_id].count("STARTED MOVING")
            stop_count = event_history[object_id].count("STOPPED MOVING")
            total_events = len(event_history[object_id])

            #cv2.putText(
             #   output,
              #  f"Events: {total_events} | Starts: {start_count} | Stops: {stop_count}",
               # (int(x1), int(y2) + 165),
                #cv2.FONT_HERSHEY_SIMPLEX,
                #0.45,
                #(0, 255, 255),
                #2
            #)

            if total_events >= 4:
                if start_count >= 2 and stop_count >= 2:
                    pattern = "REPEATED MOVEMENT"
                else:
                    pattern = "IRREGULAR"
            else:
                pattern = "INSUFFICIENT DATA"

            cv2.putText(
                output,
                f"Pattern: {pattern}",
                (int(x1), int(y2) + 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )

        # --------------------------------
        # OBJECT RELATIONSHIPS
        # --------------------------------

        if results[0].boxes.id is not None:

            centers = []

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, object_id in zip(boxes, ids):

                x1, y1, x2, y2 = box

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                centers.append(
                    (int(object_id), center_x, center_y)
                )

            for i in range(len(centers)):

                for j in range(i + 1, len(centers)):

                    id1, x1, y1 = centers[i]
                    id2, x2, y2 = centers[j]

                    distance = math.sqrt(
                        (x2 - x1) ** 2 +
                        (y2 - y1) ** 2
                    )

                    if distance < 150:
                        relation = "NEAR"
                    else:
                        relation = "FAR"

                    # Relative horizontal position
                    if x1 < x2:
                        position_relation = "LEFT"
                    else:
                        position_relation = "RIGHT"

                    scene_relationships.append(
                        (id1, id2, relation, position_relation)
                    )

        # --------------------------------
        # DISPLAY SCENE RELATIONSHIPS
        # --------------------------------

        relationship_y = 85

        for id1, id2, relation, position_relation in scene_relationships:

            cv2.putText(
                output,
                f"ID {id1} <-> ID {id2}: {relation}, {position_relation}",
                (20, relationship_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            relationship_y += 25

        # =========================
        # UPDATE 3D TRAJECTORY
        # =========================

        ax.clear()

        # --------------------------------
        # DRAW SPARSE 3D POINT CLOUD
        # --------------------------------

        if len(point_cloud) > 0:

            cloud_x = [point[0] for point in point_cloud]
            cloud_y = [point[1] for point in point_cloud]
            cloud_z = [point[2] for point in point_cloud]

            ax.scatter(
                cloud_x,
                cloud_y,
                cloud_z,
                s=5
            )

        for tracked_id, points in spatial_tracks.items():

            if tracked_id not in active_ids:
                continue
            if len(points) < 2:
                continue

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            zs = [point[2] for point in points]

            ax.plot(
                xs,
                ys,
                zs,
                marker="o",
                label=f"Object {tracked_id}"
            )

            ax.scatter(
                xs[-1],
                ys[-1],
                zs[-1],
                s=80,
                label=f"Current {tracked_id}"
            )




        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title("CHRONOS - 3D Object Trajectory")

        if spatial_tracks:
            ax.legend()

        plt.pause(0.001)
        
        # --------------------------------
        # CHRONOS CAMERA DASHBOARD
        # --------------------------------

        dashboard_x = frame.shape[1] - 285
        dashboard_y = 75
        dashboard_w = 270
        dashboard_h = 435

        # Dark dashboard background
        overlay = output.copy()

        cv2.rectangle(
            overlay,
            (dashboard_x, dashboard_y),
            (dashboard_x + dashboard_w, dashboard_y + dashboard_h),
            (20, 20, 20),
            -1
        )

        # Make dashboard slightly transparent
        cv2.addWeighted(
            overlay,
            0.80,
            output,
            0.20,
            0,
            output
        )

        # Dashboard border
        cv2.rectangle(
            output,
            (dashboard_x, dashboard_y),
            (dashboard_x + dashboard_w, dashboard_y + dashboard_h),
            (255, 255, 0),
            2
        )

        # Dashboard title
        cv2.putText(
            output,
            "CHRONOS DASHBOARD",
            (dashboard_x + 15, dashboard_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            output,
            "SPATIAL MOTION INTELLIGENCE",
            (dashboard_x + 15, dashboard_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1
        )

        # Active object count
        cv2.putText(
            output,
            f"ACTIVE OBJECTS: {len(active_ids)}",
            (dashboard_x + 15, dashboard_y + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            2
        )

        # Current object information
        cv2.putText(
            output,
            f"OBJECT ID: {object_id}",
            (dashboard_x + 15, dashboard_y + 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"STATUS: {status}",
            (dashboard_x + 15, dashboard_y + 133),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"DIRECTION: {direction}",
            (dashboard_x + 15, dashboard_y + 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"SPEED: {speed:.2f} px/frame",
            (dashboard_x + 15, dashboard_y + 183),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"DEPTH: {depth_value:.2f}",
            (dashboard_x + 15, dashboard_y + 208),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"XYZ: ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})",
            (dashboard_x + 15, dashboard_y + 233),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"3D SPEED: {speed_3d:.4f}",
            (dashboard_x + 15, dashboard_y + 258),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"PATH: {path_type}",
            (dashboard_x + 15, dashboard_y + 283),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"BEHAVIOR: {behavior}",
            (dashboard_x + 15, dashboard_y + 308),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"PREDICTION: ({int(predicted_x)}, {int(predicted_y)})",
            (dashboard_x + 15, dashboard_y + 333),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 0, 255),
            1
        )   
        cv2.putText(
            output,
            f"DOMINANT DIRECTION: {dominant_direction}",
            (dashboard_x + 15, dashboard_y + 358),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            output,
            f"PREDICTION CONFIDENCE: {prediction_confidence:.1f}%",
            (dashboard_x + 15, dashboard_y + 383),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1
        ) 

        cv2.putText(
            output,
            f"TEMPORAL STATE: {temporal_state}",
            (dashboard_x + 15, dashboard_y + 408),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1
        )

    # Show camera
    cv2.imshow(
        "CHRONOS - Motion Prediction",
        output
    )




    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release camera
camera.release()

# Close windows
cv2.destroyAllWindows()

plt.ioff()
plt.close()
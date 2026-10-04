# CHRONOS

## Spatial Motion Intelligence & Future Trajectory Prediction

CHRONOS is a vision-based spatial intelligence system that observes an environment, detects and tracks objects, estimates scene depth, reconstructs a sparse 3D representation, analyzes object motion, understands spatial relationships, and predicts likely future trajectories.

Instead of treating a camera feed as a sequence of independent frames, CHRONOS maintains information about the **past, present, and predicted future** of objects within a scene.

> **PAST → PRESENT → FUTURE**

---

##  Key Capabilities

- Real-time object detection using YOLO
- Persistent object tracking with unique IDs
- ORB-based feature detection and matching
- Camera motion estimation
- Monocular depth estimation using MiDaS
- Grayscale depth-map visualization
- Sparse 3D point-cloud reconstruction
- 3D object position estimation
- 2D and 3D trajectory tracking
- Object motion and speed analysis
- Direction and movement-state analysis
- Temporal movement history
- Scene relationship understanding
- Relative spatial relationships such as:
  - NEAR / FAR
  - LEFT / RIGHT
- Future trajectory estimation
- Prediction confidence estimation
- Real-time visualization dashboard

---

##  System Overview

CHRONOS follows a multi-stage computer vision pipeline:

```text
                    CAMERA INPUT
                         │
                         ▼
                OBJECT DETECTION
                     (YOLO)
                         │
                         ▼
                  OBJECT TRACKING
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        MOTION ANALYSIS       ORB FEATURES
              │                     │
              │                     ▼
              │              CAMERA MOTION
              │                ESTIMATION
              │
              ▼
       MiDaS DEPTH ESTIMATION
              │
              ▼
       SPARSE 3D RECONSTRUCTION
              │
              ▼
        SCENE UNDERSTANDING
              │
              ▼
       TEMPORAL MOTION HISTORY
              │
              ▼
       FUTURE TRAJECTORY
          PREDICTION
              │
              ▼
        CHRONOS DASHBOARD
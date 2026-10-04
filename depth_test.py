import cv2
import torch

# Load MiDaS depth model
model_type = "MiDaS_small"

midas = torch.hub.load(
    "intel-isl/MiDaS",
    model_type
)

midas.eval()

# Load transformation
midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.small_transform

# Open webcam
camera = cv2.VideoCapture(0)

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not access camera")
        break

    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare image for MiDaS
    input_batch = transform(img)

    # Predict depth
    with torch.no_grad():
        prediction = midas(input_batch)

    # Resize depth prediction
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

    # Convert to numpy
    depth_map = prediction.cpu().numpy()

    # Normalize depth for display
    depth_display = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = depth_display.astype("uint8")

    # Convert depth map to 3-channel image
    depth_display = cv2.cvtColor(
        depth_display,
        cv2.COLOR_GRAY2BGR
    )

    # Resize both images to the same size
    camera_view = cv2.resize(
        frame,
        (640, 480)
    )
    depth_display = cv2.resize(
        depth_display,
        (640, 480)
    )

    # Put camera and depth side by side
    combined = cv2.hconcat([
        camera_view,
        depth_display
    ])

    # Show both
    cv2.imshow(
        "CHRONOS - Camera + Depth",
        combined
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

import os
import cv2
from ultralytics import YOLO


def get_yolov8(vidoe_path):
    cap = cv2.VideoCapture(vidoe_path)
    currentframe = 0
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            currentframe += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


dataset_path = 'D:/side_project/DatasetB/videos'

person_carry = ["bg", "cl", "nm", ]
camera_angle = ["000", "018", "036", "054", "072",
                "090", "108", "126", "144", "162", "182"]

# Load a model
model = YOLO("../models/yolov8x-seg.pt")

for person_id in range(1, 84 + 1):
    for walking_count in range(1, 2 + 1):
        print(person_id)
        print(walking_count)

        video_file = f"{str(person_id).zfill(3)}-{person_carry[0]}-{str(walking_count).zfill(2)}-{camera_angle[0]}.avi"
        vidoe_path = dataset_path + "/" + video_file
        if os.path.isfile(vidoe_path):
            print(vidoe_path)
            get_yolov8(vidoe_path)

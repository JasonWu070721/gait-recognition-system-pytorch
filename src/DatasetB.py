


import os

dataset_path = 'D:/side_project/DatasetB/videos'

person_carrys = ["bg-01", "bg-02", "cl-01", "cl-02", "nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06"]
camera_angles = ["000", "018", "036", "054", "072",
                "090", "108", "126", "144", "162", "180"]

video_count = 0

for person_id in range(1, 84 + 1):
    for person_carry in person_carrys:
        for camera_angle in camera_angles:
            video_file = f"{str(person_id).zfill(3)}-{person_carry}-{camera_angle}.avi"
            vidoe_path = dataset_path + "/" + video_file
            print(vidoe_path)
            if os.path.isfile(vidoe_path):
                
                video_count += 1
                print("video_count: ", video_count)
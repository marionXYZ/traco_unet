import cv2
import json
import os
import numpy as np
from pathlib import Path


def save_frames_and_masks(path, dest_path, image_shape, num_frames_per_video):
    """
    This function extracts frames from the videos and creates binary masks from the annotations.
    Both frames and mask are resized and saved as PNG files.
    """
    for vid in os.listdir(path):
        path = Path(path)
        if ".mp4" in vid:
            with open(path / vid.replace("mp4", "traco")) as f:
                annotations = json.load(f)['rois']

            cap = cv2.VideoCapture(str(path / vid))
            ret, frame = cap.read()
            org_shape = frame.shape

            z = 0  # frame counter
            while ret:
                mask = np.zeros(shape=image_shape)
                for annot in annotations:
                    if annot['z'] == z:
                        # Get pos and scale it down to fit the target_shape
                        pos = annot['pos']
                        pos[0] = int(pos[0] * image_shape[0] // org_shape[1])
                        pos[1] = int(pos[1] * image_shape[1] // org_shape[0])

                        # Set the position if the Hexbug in the binary mask to 1 (single pixel)
                        # mask[int(pos[1]), int(pos[0])] = (255, 255, 255)

                        # Create a mask for the hexbug's head (small circle at head position)
                        mask = cv2.circle(
                            mask,
                            center=(pos[0], pos[1]),
                            radius=4,
                            color=(255, 255, 255),
                            thickness=-1
                        )

                # Resize the frame to the target size using bilinear interpolation
                resized_frame = cv2.resize(frame, image_shape, interpolation=cv2.INTER_LINEAR)

                # Save to disk
                filename = Path(vid).stem
                cv2.imwrite(f"{dest_path}/{filename}_{z}_img.png", resized_frame)
                cv2.imwrite(f"{dest_path}/{filename}_{z}_mask.png", mask)

                ret, frame = cap.read()  # read next frame
                z += 1  # increase frame counter

                if z >= num_frames_per_video:
                    break


os.mkdir("train_data")
os.mkdir("recorded_data")
save_frames_and_masks("../training", "train_data", (128, 128), num_frames_per_video=50)
save_frames_and_masks("recorded_video", "recorded_data", (128, 128), num_frames_per_video=100)

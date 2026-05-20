
import cv2
import os
import time

def read_webcam(capture_dir="captured_images"):
    """
    Always streams webcam. When 'c' is pressed, captures one image to a folder and stops capturing (but continues streaming). Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    print("Press 'n' to save to folder '1', 'b' to save to folder '0', 'q' to quit.")
    img_count_1 = 0
    img_count_0 = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            folder = os.path.join(capture_dir, "1")
            if not os.path.exists(folder):
                os.makedirs(folder)
            img_path = os.path.join(folder, f"frame_{img_count_1:05d}.jpg")
            resized_frame = cv2.resize(frame, (32, 32))
            cv2.imwrite(img_path, resized_frame)
            print(f"Captured {img_path} (folder 1)")
            img_count_1 += 1
        elif key == ord('b'):
            folder = os.path.join(capture_dir, "0")
            if not os.path.exists(folder):
                os.makedirs(folder)
            img_path = os.path.join(folder, f"frame_{img_count_0:05d}.jpg")
            resized_frame = cv2.resize(frame, (32, 32))
            cv2.imwrite(img_path, resized_frame)
            print(f"Captured {img_path} (folder 0)")
            img_count_0 += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    read_webcam(capture_dir="captured_images")

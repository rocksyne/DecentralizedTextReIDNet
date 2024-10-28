import cv2
from mtcnn import MTCNN
import numpy as np

def blur_faces_in_stream():
    # Load the MTCNN face detector
    detector = MTCNN()

    # Start the video capture from the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Unable to open the camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera")
            break

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Blur each detected face
        for face in faces:
            x, y, width, height = face['box']
            x, y = abs(x), abs(y)

            # Extract the region of the frame that contains the face
            face_region = frame[y:y+height, x:x+width]

            # Apply a Gaussian blur to the face region
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)

            # Replace the original face region with the blurred one
            frame[y:y+height, x:x+width] = blurred_face

        # Display the resulting frame
        cv2.imshow('Face Blurring', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Run the function
blur_faces_in_stream()

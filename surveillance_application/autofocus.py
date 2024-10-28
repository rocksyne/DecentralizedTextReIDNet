import cv2
import subprocess

#Disable autofocus
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=focus_automatic_continuous=0"])

#Set focus manually (adjust the value as needed)
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=focus_absolute=0"])


# OpenCV capturing
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

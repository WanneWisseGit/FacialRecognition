import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.imwrite("SVM_images/wiggert" + str(count) + ".jpg", frame)
    count = count + 1
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
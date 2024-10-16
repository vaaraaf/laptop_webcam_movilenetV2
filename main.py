import time
import cv2
from evaluation import evaluate_image
import git
import cv2
from pathlib import Path

file_path = Path(__file__)
parent_path = file_path.parent
images_path = parent_path / 'images'
sample_image_path = images_path / 'sample.png'
try:
    from ToolBox_Pytorch import download_model
except:
    print('Downloading repo from github')
    git.Git(parent_path).clone('https://github.com/vaaraaf/ToolBox_Pytorch')
    from ToolBox_Pytorch import download_model
cap = cv2.VideoCapture(0)
my_model, my_transform = download_model.download_mobilenet_v2()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow('Webcam Feed', frame)

    if ret:
        cv2.imwrite(sample_image_path, frame)
        evaluate_image(image_path=sample_image_path,
                       model=my_model,
                       transform=my_transform)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
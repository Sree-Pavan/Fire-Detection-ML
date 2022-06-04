import cv2 as cv2
import tensorflow as tf
import numpy as np
from PIL import Image

feed = cv2.VideoCapture(0)
lower_bound = np.array([0, 0, 255])
upper_bound = np.array([90, 255, 255])
model = tf.keras.models.load_model('InceptionV3.h5')

while True:
    ret, frame = feed.read()
    # cv2.imshow("Fire Detection", frame)
    frame = cv2.resize(frame, (1280, 720))
    frame_smooth = cv2.GaussianBlur(frame, (7, 7), 0)
    mask = np.zeros_like(frame)
    mask[0:720, 0:1280] = [255, 255, 255]
    img_roi = cv2.bitwise_and(frame_smooth, mask)
    frame_hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
    image_binary = cv2.inRange(frame_hsv, lower_bound, upper_bound)
    # cv2.imshow("Fire Detection", frame)
    cv2.imshow("Fire Detection", image_binary)
    check_if_fire_detected = cv2.countNonZero(image_binary)
    threshold = 5000
    if int(check_if_fire_detected) >= threshold:
        # contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0] if len(contours) == 2 else contours[1]
        # for cntr in contours:
        #     x, y, w, h = cv2.boundingRect(cntr)
        #     cv2.rectangle(image_binary, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.putText(image_binary, "Fire Detected !", (300, 60), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)
        im = Image.fromarray(frame, 'RGB')
        im = im.resize((224, 224))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        # Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        # if prediction is 0, which means there is fire in the frame.
        if prediction == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            print("Fire Probability" + str(probabilities[prediction]))
            print()

    cv2.imshow("Fire Detection", image_binary)

    if cv2.waitKey(10) == 27:
        break
# img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 
# img_cap = plt.imshow(img_RGB)


feed.release()
cv2.destroyAllWindows()

import urllib.request
import cv2
import numpy as np

url = 'http://100.123.102.255/cam-hi.jpg'
#cv2.namedWindow("live cam testing", cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(url)

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)

    cv2.imshow('live cam testing', im)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


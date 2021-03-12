import cv2
import time
import numpy as np

"""
The structure of the neural network is stored as a .prototxt file
The weights of the layers of the neural network are stored as a .caffemodel file
"""
# Load Model and Image
protoFile = "Hand/pose_deploy.prototxt"
weightsFile = "Hand/pose_iter_102000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

nPoints = 22
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

frame = cv2.imread("Image/img3.jpg")
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
aspect_ratio = frameWidth / frameHeight

threshold = 0.1

t = time.time()
# input image dimensions for the network
inHeight = 468
inWidth = int(aspect_ratio * inHeight)
# Prepare input format with inpBlob
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
# It returns 4-dimensional Mat with NCHW dimensions order.
# "NCHW" means a data whose layout is (batch_size, channel, height, width)

net.setInput(inpBlob)
output = net.forward()

print("time taken by network : {:.3f}".format(time.time() - t))

# Empty list to store the detected key-points
points = []
# Show Detections
"""
The output has 22 matrices with each matrix being the Probability Map of a keypoint. Given below is the 
probability heatmap superimposed on the original image for the keypoint belonging to the tip of thumb and base of 
little finger. For finding the exact key-points, first, we scale the probability map to the size of the original image. 
Then find the location of the keypoint by finding the maxima of the probability map. This is done using the minmaxLoc 
function in OpenCV. We draw the detected points along with the numbering on the image. 
"""
for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    if prob > threshold:
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))
    else:
        points.append(None)

# Draw Skeleton
"""
We will use the detected points to get the skeleton formed by the 
key-points and draw them on the image.
"""
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        if int(partA) in range(0, 4) and int(partB) in range(1, 5):
            cv2.line(frame, points[partA], points[partB], (0, 128, 255), 2)
            cv2.circle(frame, points[partA], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)

        if (partA == 0 or partA == 5 or partA == 6 or partA == 7) and int(partB) in range(5, 9):
            cv2.line(frame, points[partA], points[partB], (255, 0, 255), 2)
            cv2.circle(frame, points[partA], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)

        if (partA == 0 or partA == 9 or partA == 10 or partA == 11) and int(partB) in range(9, 13):
            cv2.line(frame, points[partA], points[partB], (255, 255, 0), 2)
            cv2.circle(frame, points[partA], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)

        if (partA == 0 or partA == 13 or partA == 14 or partA == 15) and int(partB) in range(13, 17):
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)

        if (partA == 0 or partA == 17 or partA == 18 or partA == 19) and int(partB) in range(17, 21):
            cv2.line(frame, points[partA], points[partB], (255, 128, 0), 2)
            cv2.circle(frame, points[partA], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 3, (0, 0, 255), thickness=2, lineType=cv2.FILLED)


"""Result showing the skeleton obtained by joining the keypoint pairs"""

row, column, band = frame.shape
a = column//column
c = column//2
b = c//2
d = c + b
e = column
font = cv2.FONT_HERSHEY_SIMPLEX
frame = cv2.putText(frame, '1', (a, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
frame = cv2.putText(frame, '2', (b-10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
frame = cv2.putText(frame, '3', (c-15, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
frame = cv2.putText(frame, '4', (d-20, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
frame = cv2.putText(frame, '5', (e-25, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

# no 1 finger detection
if points[4]:
    frame = cv2.line(frame, (a + 5, 30), points[4], (255, 255, 255), 1, cv2.LINE_AA)
elif points[3]:
    frame = cv2.line(frame, (a + 5, 30), points[3], (255, 255, 255), 1, cv2.LINE_AA)
elif points[2]:
    frame = cv2.line(frame, (a + 5, 30), points[2], (255, 255, 255), 1, cv2.LINE_AA)

# no 2 finger detection
if points[8]:
    frame = cv2.line(frame, (b, 30), points[8], (255, 255, 255), 1, cv2.LINE_AA)
elif points[7]:
    frame = cv2.line(frame, (b, 30), points[7], (255, 255, 255), 1, cv2.LINE_AA)
elif points[6]:
    frame = cv2.line(frame, (b, 30), points[6], (255, 255, 255), 1, cv2.LINE_AA)

# no 3 finger detection
if points[12]:
    frame = cv2.line(frame, (c, 30), points[12], (255, 255, 255), 1, cv2.LINE_AA)
elif points[11]:
    frame = cv2.line(frame, (c, 30), points[11], (255, 255, 255), 1, cv2.LINE_AA)
elif points[10]:
    frame = cv2.line(frame, (c, 30), points[10], (255, 255, 255), 1, cv2.LINE_AA)

# no 4 finger detection
if points[16]:
    frame = cv2.line(frame, (d, 30), points[16], (255, 255, 255), 1, cv2.LINE_AA)
elif points[15]:
    frame = cv2.line(frame, (d, 30), points[15], (255, 255, 255), 1, cv2.LINE_AA)
elif points[14]:
    frame = cv2.line(frame, (d, 30), points[14], (255, 255, 255), 1, cv2.LINE_AA)

# no 5 finger detection
if points[20]:
    frame = cv2.line(frame, (e, 30), points[20], (255, 255, 255), 1, cv2.LINE_AA)
elif points[19]:
    frame = cv2.line(frame, (e, 30), points[19], (255, 255, 255), 1, cv2.LINE_AA)
elif points[18]:
    frame = cv2.line(frame, (e, 30), points[18], (255, 255, 255), 1, cv2.LINE_AA)

"""Display output"""

# cv2.imshow('Output-Key-points', frameCopy)
cv2.imshow('Output-Skeleton', frame)

# cv2.imwrite('Output-Key-points.jpg', frameCopy)
cv2.imwrite('Output-Skeleton.jpg', frame)

print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)

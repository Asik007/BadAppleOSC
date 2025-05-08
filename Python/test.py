    
import cv2
import numpy as np

scanNumPF = 2; # Number of scans per frame
Fs = 48000; #Sampling rate

dotNumPF = round(Fs/60); # Points per frame
dotNum = round(dotNumPF/scanNumPF); # Points per scan


frame = cv2.imread(r'./Test_Vid/Bad Apple (Img)/0060.png')

WHR = frame.shape[1] / frame.shape[0]

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (1024, int(1024 / WHR)))
blurred = cv2.GaussianBlur(resized, (3, 3), 1024 / dotNum)
_, otsued = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
edges = cv2.Canny(np.uint8(otsued), 100, 200)

cv2.imshow('edges', edges)
cv2.imshow('resized', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
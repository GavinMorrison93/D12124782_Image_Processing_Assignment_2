# Gavin Morrison - D12124782 - DT211C-4

# 1. I inputted the image and converted it to Greyscale and HSV for the 
# operations within the program.

# 2. I created a Numpy array of the upper and lower red values of our 
# HSV image

# 3. I created a mask of the red values

# 4. Red content of the image is isolated via a Bitwise 'And' operation

# 5. The Greyscale image is inverted

# 6. A Binary threshold is applied to our inverted Greyscale image.

# 7. The thresholded image is then submitted to another bitwise 
# inversion which produces the white content of the image.

# 8. The isolated red and white content are combined.

# 9. The red and white components of the original image are dilated to 
# produce a slight overlap allowing the detection of connected regions.

# 10. The dilated images are 'Anded' to produce an image of only the 
# overlapping regions.

# 11. Canny is applied to detect edges within the image of connected regions.

# 12. A Hough Line Transform is applied to the results of Canny to try and 
# find the horizontal lines within Wally's sweater.

# 13. Wally is located. Please see comments throughout the code for further 
# step by step details.

# I initially tried using Contours to locate Wally but I had very limited 
# results owing to the low resolution of Wally within the overall image.

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image

# Original image is imported
originalImage = cv2.imread("Where.jpg")

# Original image is converted to greyscale for white segmentation.
originalImageGrey = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

# Original image is converted to a HSV Image for red segmentation.
HSVImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

# A Numpy array is created of upper and lower red values.
lowerRedValue = np.array([139,100,100]) 
upperRedValue = np.array([189,255,255])

# Mask is created of all red values.
redMask = cv2.inRange(HSVImage, lowerRedValue, upperRedValue)

# The red content of the image is isolated.
redContent = cv2.bitwise_and(originalImage,originalImage, mask= redMask)

# The Greyscale image is inverted via bitwise.
greyInverted = cv2.bitwise_not(originalImageGrey)

# A binary threshold is applied to the inverted greyscale image.
T = 20
T, binaryInvertedGreyscale = cv2.threshold(greyInverted, thresh = T, maxval = 255,
type = cv2.THRESH_BINARY)

# This bitwise not operation creates the white content image.
whiteContent = cv2.bitwise_not(binaryInvertedGreyscale)

# white and red segmented images are written to file and the images inputed.
cv2.imwrite("white.png", whiteContent)
cv2.imwrite("red.png", redContent)
redImage = cv2.imread("red.png")
whiteImage = cv2.imread("white.png")

#The red and white images are combined together and written to file.
redWhiteCombined = redImage + whiteImage
cv2.imwrite("combined.png", redWhiteCombined)

# Dilation is perfomed on red and white images to create a slight overlap to allow the location of where red and white are connected.
whiteKernel = np.ones((2,1), np.uint8)
redKernel = np.ones((2,1), np.uint8)
whiteDilation = cv2.dilate(whiteContent, whiteKernel, iterations=1)
redDilation = cv2.dilate(redMask, redKernel, iterations=1)

# Dilated images are combined in 'And' operation which isolates connecting areas between red and white
connectedRegions = whiteDilation & redDilation

# Connected regions written to file and then inputed 
cv2.imwrite("connectedregions.png", connectedRegions)
connectedRegionsImage = cv2.imread("connectedregions.png")

# The connected regions are converted to greyscale. 
connectedGreyscale = cv2.cvtColor(connectedRegionsImage, cv2.COLOR_BGR2GRAY)

# The edges are located on the connected regions.
connectedEdges = cv2.Canny(connectedGreyscale, 500, 300)

# A Hough Line Transform is applied that scans the image for lines in the hope that 
# it will locate the stripes on Wally's sweater.
houghLines = cv2.HoughLinesP(connectedEdges, 1, np.pi/90, 30, maxLineGap=2)

#For loop plotting location of lines found in image.
for line in houghLines:
    x1, y1, x2, y2 = line[0]
    cv2.line(connectedRegionsImage, (x1, y1), (x2, y2), (255, 255, 255), 90)

# Location written to file and inputed	
cv2.imwrite('wallylocation.png',connectedRegionsImage)
wallyLocation = cv2.imread("wallylocation.png")

# Location found is combined with original image in 'And' operation and written to file.
wallyFound = wallyLocation & originalImage
cv2.imwrite('wallyfound.png',wallyFound)

# Colour space conversions for matPlotLib display.
OImatPlotLib = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
RCmatPlotLib = cv2.cvtColor(redContent, cv2.COLOR_BGR2RGB)
RWmatPlotLib = cv2.cvtColor(redWhiteCombined, cv2.COLOR_BGR2RGB)
WFmatPlotLib = cv2.cvtColor(wallyFound, cv2.COLOR_BGR2RGB)

# A display of the result of our operations
plt.subplot(231), plt.imshow(OImatPlotLib, cmap ='gray'), plt.title( 'Original Image' )
plt.subplot(232), plt.imshow(whiteContent, cmap ='gray'), plt.title( 'White Content' )
plt.subplot(233), plt.imshow(RCmatPlotLib, cmap ='gray'), plt.title( 'Red Content' )
plt.subplot(234), plt.imshow(RWmatPlotLib, cmap ='gray'), plt.title( 'Red & White combined' )
plt.subplot(235), plt.imshow(connectedRegions, cmap ='gray'), plt.title( 'Connected Regions' )
plt.subplot(236), plt.imshow(WFmatPlotLib, cmap ='gray'), plt.title( 'Wally Found' )
plt.show()
#resize an image

import cv2 as cv

img = cv.imread("pictures/cat0.jpg")
cv.imshow("Original_image",img)

def resize_frame(img,scale_width = 0.75,scale_height = 0.75):
    width = int(img.shape[1]*scale_width)
    height = int(img.shape[0]*scale_height)
    dim = (width,height)
    return cv.resize(img,dim,interpolation=cv.INTER_AREA)
img = resize_frame(img,1, 0.5)
cv.imshow("Resized_image",img)
# img = resize_frame(img,0.5,0.5)
# cv.imshow("0.5_image",img)
cv.waitKey(0)
cv.destroyAllWindows()
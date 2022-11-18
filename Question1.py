#take all the files(img) from a folder
#reduce the size of the image
#save the image in another folder
#change the colorful image to black and white

import cv2 as cv
import os
import shutil
import matplotlib.pyplot as plt

# path = "pictures"
# if os.path.exists(path):
#     shutil.rmtree(path)
# os.makedirs(path)
# img = cv.imread("pic.jpg")
# for i in range(100):
#     cv.imwrite(os.path.join(path, "cat"+str(i)+".jpg"),img)
path = "pictures/"
images = []
for file in os.listdir(path):
    img = cv.imread(os.path.join(path, file))
    images.append(img)
# show_img = images[0]
# plt.imshow(cv.cvtColor(show_img, cv.COLOR_BGR2RGB))
# plt.show()
upper = 0
lower = 800
left = 1000
right = 1750
new_path = "copy_pictures"
if os.path.exists(new_path):
    shutil.rmtree(new_path)
os.makedirs(new_path)
num = 0
for i in images:
    img = i[upper:lower, left:right]
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite(os.path.join(new_path, "copy_cat"+str(num)+".jpg"),img)
    num += 1
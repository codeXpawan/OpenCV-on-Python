import cv2 as cv
import numpy as np



def find_object(image,tuning_params):
    blur = 2
    x_min = tuning_params["x_min"]
    y_min = tuning_params["y_min"]
    x_max = tuning_params["x_max"]
    y_max = tuning_params["y_max"]
    
    search_window = [x_min, y_min, x_max, y_max]

    working_image    = cv.blur(image, (blur, blur))
    # cv.imshow('blur_image', working_image)
    if search_window is None: search_window = [0.0, 0.0, 1.0, 1.0]
    search_window_px = convert_rect_perc_to_pixels(search_window, image)
    #- Convert image from BGR to HSV
    working_image     = cv.cvtColor(working_image, cv.COLOR_BGR2HSV)    
    
    #- Apply HSV threshold
    thresh_min_1 = (tuning_params["h_min"], tuning_params["s_min"], tuning_params["v_min"])
    thresh_max_1 = (tuning_params["h_max"], tuning_params["s_max"], tuning_params["v_max"])
    working_image_1    = cv.inRange(working_image, thresh_min_1, thresh_max_1)
    # Dilate and Erode
    working_image_1 = cv.dilate(working_image_1, None, iterations=2)
    working_image_1 = cv.erode(working_image_1, None, iterations=2)
    
    tuning_image_1 = cv.bitwise_and(image,image,mask = working_image_1)
    #removing noise 
    kernel = np.ones((3,3),np.uint8)
    working_image_1 = cv.morphologyEx(working_image_1, cv.MORPH_OPEN, kernel, iterations=1)
    # Invert the image to suit the blob detector
    working_image_1 = 255-working_image_1
    #Morphology EX
    # working_image = cv.morphologyEx(working_image, cv.MORPH_OPEN, kernel)
    working_image_1 = cv.morphologyEx(working_image_1, cv.MORPH_CLOSE, kernel)
    # Set up the SimpleBlobdetector with default parameters.
    params = cv.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0
    params.maxCircularity = 1
    
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 1000
        
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 2000000
    detector = cv.SimpleBlobDetector_create(params)
    
    # detector = cv.SimpleBlobDetector_create(params)

    # Run detection!
    # cv.imshow('working_image',working_image)
    # canny = cv.Canny(working_image,100,150)
    # contours, hierarchies  = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # # cv.imshow("Canny",canny)
    # contours_poly = [None]*len(contours)
    # area = [None]*len(contours)
    # boundRect = [None]*len(contours)
    # text = [None]*len(contours)
    # for i, c in enumerate(contours):
    #     contours_poly[i] = cv.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv.minAreaRect(contours_poly[i])
    #     area[i] = cv.contourArea(contours_poly[i])
    #     text = cv.putText(image, str(area[i]), (int(boundRect[i][0][0]), int(boundRect[i][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (45, 255, 166), 1)
    # for i in range(len(contours)):
    #     box = cv.boxPoints(boundRect[i])
    #     box = np.int0(box)
    #     print(area[i])
    #     if area[i]>4000:
    #         cv.drawContours(image,[box],0,(0,0,255),2)
    #         # cv.putText(img,str(area[i]),(int(boundRect[i]),int(box[0][1]/2)),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    #         # print(contours_poly[i])
    #     # cv.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])),(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,255,0), 2)
    #     else:
    #         cv.drawContours(image, [box], 0, (0,255,0), 2)
    #         # cv.putText(img,str(area[i]),(int(box[0][1]/2),int(box[0][1]/2)),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    # # cv.imshow("Rectangle",image)

    keypoints = detector.detect(working_image_1)

    size_min_px = tuning_params['sz_min']*working_image_1.shape[1]/100.0
    size_max_px = tuning_params['sz_max']*working_image_1.shape[1]/100.0

    keypoints = [k for k in keypoints if k.size > size_min_px and k.size < size_max_px]

    
    # Set up main output image
    line_color=(0,255,0)

    out_image = cv.drawKeypoints(image, keypoints, np.array([]), line_color, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_image = draw_window2(out_image, search_window_px)

    # Set up tuning output image
    
    tuning_image_1 = cv.drawKeypoints(tuning_image_1, keypoints, np.array([]), line_color, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # tuning_image = draw_window(tuning_image, search_window)
    # cv2.rectangle(image,(x_min_px,y_min_px),(x_max_px,y_max_px),color,line)
    tuning_image_1 = draw_window2(tuning_image_1, search_window_px)
    # cv.imshow('out_image',out_image)
    # cv.imshow('tuining_image',tuning_image)
    

    keypoints_normalised = [normalise_keypoint(working_image_1, k) for k in keypoints]
    # print(keypoints_normalised)

    return keypoints_normalised, out_image, tuning_image_1

    
def normalise_keypoint(cv_image, kp):
    rows = float(cv_image.shape[0])
    cols = float(cv_image.shape[1])
    # print(rows, cols)
    center_x    = 0.5*cols
    center_y    = 0.5*rows
    # print(center_x)
    x = (kp.pt[0] - center_x)/(center_x)
    y = (kp.pt[1] - center_y)/(center_y)
    return cv.KeyPoint(x, y, kp.size/cv_image.shape[1])

    
def convert_rect_perc_to_pixels(rect_perc, image):
    rows = image.shape[0]
    cols = image.shape[1]

    scale = [cols, rows, cols, rows]

    
    # x_min_px    = int(cols*window_adim[0])
    # y_min_px    = int(rows*window_adim[1])
    # x_max_px    = int(cols*window_adim[2])
    # y_max_px    = int(rows*window_adim[3]) 
    return [int(a*b/100) for a,b in zip(rect_perc, scale)]

def draw_window2(image,              #- Input image
                rect_px,        #- window in adimensional units
                color=(255,0,0),    #- line's color
                line=5,             #- line's thickness
               ):
    
    #-- Draw a rectangle from top left to bottom right corner

    return cv.rectangle(image,(rect_px[0],rect_px[1]),(rect_px[2],rect_px[3]),color,line)



# upper_frame = [0,240,0,640]
# lower_frame = [240,480,0,640]
# color_range = [np.array([96,100,100]),np.array([116,255,255])]

def detect_object(frame,frame_size,color_range):
    sliced_frame = frame[frame_size[0]:frame_size[1],frame_size[2]:frame_size[3]]
    sliced_hsv_frame = cv.cvtColor(sliced_frame,cv.COLOR_BGR2HSV)
    cv.imshow("hsv_frame",sliced_hsv_frame)
    mask = cv.inRange(sliced_hsv_frame,color_range[0],color_range[1])
    filter = cv.bitwise_and(sliced_hsv_frame,sliced_hsv_frame,mask = mask)
    cv.imshow('filter',filter)

def get_tuning_params(objects):
    trackbar_names_tuning = ["x_min","x_max","y_min","y_max","sz_min","sz_max"]
    trackbar_names_objects = ["h_min","h_max","s_min","s_max","v_min","v_max"]
    params = {}
    for key in trackbar_names_tuning:
        params[key] = cv.getTrackbarPos(key, "Tuning")
    
    for i in range(objects):
        window_name = f"Object_{i}"
        for key in trackbar_names_objects:
            params[key+f"_{i}"] = cv.getTrackbarPos(key+f"_{i}", window_name)
    
    return params


def wait_on_gui():
    cv.waitKey(2)

def create_tuning_window(initial_values,objects):
    cv.namedWindow("Tuning", cv.WINDOW_NORMAL)
    cv.createTrackbar("x_min","Tuning",initial_values['x_min'],100,no_op)
    cv.createTrackbar("x_max","Tuning",initial_values['x_max'],100,no_op)
    cv.createTrackbar("y_min","Tuning",initial_values['y_min'],100,no_op)
    cv.createTrackbar("y_max","Tuning",initial_values['y_max'],100,no_op)
    cv.createTrackbar("sz_min","Tuning",initial_values['sz_min'],100,no_op)
    cv.createTrackbar("sz_max","Tuning",initial_values['sz_max'],100,no_op)
    for i in range (objects):
        window_name = f"Object_{i}"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.createTrackbar(f"h_min_{i}",window_name,initial_values[f'h_min_{i}'],180,no_op)
        cv.createTrackbar(f"h_max_{i}",window_name,initial_values[f'h_max_{i}'],180,no_op)
        cv.createTrackbar(f"s_min_{i}",window_name,initial_values[f's_min_{i}'],255,no_op)
        cv.createTrackbar(f"s_max_{i}",window_name,initial_values[f's_max_{i}'],255,no_op)
        cv.createTrackbar(f"v_min_{i}",window_name,initial_values[f'v_min_{i}'],255,no_op)
        cv.createTrackbar(f"v_max_{i}",window_name,initial_values[f'v_max_{i}'],255,no_op)
    
    


def no_op(x):
    pass

if __name__ == '__main__':
    image = cv.imread('./pictures/yantra.webp')
    cv.imshow("image",image)
    # if video is None or not video.isOpened():
    #     print("Cannot open Camera")
    #     exit(0)
    # while True:
    #     ret, frame = video.read()
    #     if not ret:
    #         print("Cannot receive frame")
    #         break
    #     # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #     cv.imshow('frame', frame)
    #     # print(frame.shape)
    #     # search_frame = search_window(200,300,200,400,frame)
    #     # # print(search_frame.shape)
    #     # search_frame = frame[240:480,0:640]
    #     # cv.imshow('search_frame',search_frame)
    #     # detect_object(frame, upper_frame,color_range)
        
    tuning_params = {
        'x_min': 0,
        'x_max': 1,
        'y_min': 0,
        'y_max': 1,
        'h_min': 96,
        'h_max': 116,
        's_min': 100,
        's_max': 255,
        'v_min': 100,
        'v_max': 255,
        'sz_min': 0,
        'sz_max': 100,
        'blur' : 2
    }
    keypoints,out_img,tuning_img =find_object(image,tuning_params)
    
    cv.imshow("out_img",out_img)
    cv.imshow("tuning_img",tuning_img)
    print(keypoints)
    for i, kp in enumerate(keypoints):
                x = kp.pt[0]
                y = kp.pt[1]
                s = kp.size
                print(f"Pt {i}: ({x},{y},{s})")
        
    cv.waitKey(0)
    cv.destroyAllWindows() 
    

import os
import numpy as np
import dlib
import cv2
import imutils


def take_first(tup):
    return tup[0]
def take_second(tup):
    return tup[1]

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

# detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('weights/recipt_predictor4_20221011.dat')



def coner_point(image_path):
    flage = False
    try:
        img = cv2.imread(image_path)
        
        h, w, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        rect = dlib.rectangle(1, 1, int(w - 2), int(h - 2))
        shape = predictor(gray, rect)
        
        shape = shape_to_np(shape)
        list_1 = shape.tolist()
        im_ori = img.copy()
        height, width, _ = img.shape
        final_points = list_1
        if len(final_points) == 4:
            final_points = sorted(final_points, key=take_first)
            final_points = [sorted([final_points[0], final_points[1]], key=take_second),
                            sorted([final_points[2], final_points[3]], key=take_second)]
            final_points = final_points[0] + final_points[1]
            
            max_x = \
                max([[abs(final_points[0][0] - final_points[2][0])], [abs(final_points[1][0] - final_points[3][0])]])[0]
            max_y = \
                max([[abs(final_points[0][1] - final_points[1][1])], [abs(final_points[2][1] - final_points[3][1])]])[0]
            
            src_pts = np.array([final_points[0], final_points[2], final_points[3], final_points[1]],
                               dtype=np.float32)
            
            dst_pts = np.array([[0, 0], [max_x, 0], [max_x, max_y], [0, max_y]], dtype=np.float32)
            
            perspect = cv2.getPerspectiveTransform(src_pts, dst_pts)
            im_ori = cv2.warpPerspective(im_ori, perspect, [max_x, max_y])
         
            path = 'static/corner_image/result.jpg'
            img = cv2.resize(im_ori, (700, 800))
            # cv2.imshow('asd', img)
            # cv2.waitKey(0)
            cv2.imwrite(path,im_ori)
            flage = True
            return flage, path
    except:
        return flage, {}
        

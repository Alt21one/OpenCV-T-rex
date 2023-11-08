import cv2 
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import os
import math
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#need_img = r'C:\Users\Alt21\Documents\Python_Projects\OpenCV_Test_2_Real_time\Screenshot_22.png'
#base_dir = r'C:\Users\Alt21\Documents\Python_Projects\OpenCV_Test_2_Real_time\Training'
#positive_dir = os.path.join(base_dir, 'positive')
#negative_dir = os.path.join(base_dir, 'negative')

class Vision:

    # properties
    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None

    # constructor
    def __init__(self, needle_img_path, method=cv2.TM_CCOEFF_NORMED):
        if needle_img_path:
            # load the image we're trying to match
            # https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
            self.needle_img = cv2.imread(needle_img_path, cv2.IMREAD_UNCHANGED)

            # Save the dimensions of the needle image
            self.needle_w = self.needle_img.shape[1]
            self.needle_h = self.needle_img.shape[0]

        # There are 6 methods to choose from:
        # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
        self.method = method

    def find(self, haystack_img, threshold=0.5, debug_mode=None):
        # run the OpenCV algorithm
        if self.needle_img.shape[2] == 4:
            needle_mask = self.needle_img[:, :, 3]
            self.needle_img = self.needle_img[:, :, :3]
            result = cv2.matchTemplate(haystack_img, self.needle_img, self.method, mask=needle_mask)
        else:
            result = cv2.matchTemplate(haystack_img, self.needle_img, self.method)

        # Get the all the positions from the match result that exceed our threshold
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))
        #print(locations)

        # You'll notice a lot of overlapping rectangles get drawn. We can eliminate those redundant
        # locations by using groupRectangles().
        # First we need to create the list of [x, y, w, h] rectangles
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            # Add every box to the list twice in order to retain single (non-overlapping) boxes
            rectangles.append(rect)
            rectangles.append(rect)
        # Apply group rectangles.
        # The groupThreshold parameter should usually be 1. If you put it at 0 then no grouping is
        # done. If you put it at 2 then an object needs at least 3 overlapping rectangles to appear
        # in the result. I've set eps to 0.5, which is:
        # "Relative difference between sides of the rectangles to merge them into a group."
        rectangles, weights = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
        #print(rectangles)

        points = []
        if len(rectangles):
            #print('Found needle.')

            line_color = (0, 255, 0)
            line_type = cv2.LINE_4
            marker_color = (255, 0, 255)
            marker_type = cv2.MARKER_CROSS

            # Loop over all the rectangles
            for (x, y, w, h) in rectangles:

                # Determine the center position
                center_x = x + int(w/2)
                center_y = y + int(h/2)
                # Save the points
                points.append((center_x, center_y))

                if debug_mode == 'rectangles':
                    # Determine the box position
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    # Draw the box
                    cv2.rectangle(haystack_img, top_left, bottom_right, color=line_color, 
                                lineType=line_type, thickness=2)
                elif debug_mode == 'points':
                    # Draw the center point
                    cv2.drawMarker(haystack_img, (center_x, center_y), 
                                color=marker_color, markerType=marker_type, 
                                markerSize=40, thickness=2)

        if debug_mode:
            cv2.imshow('Matches', haystack_img)
            #cv.waitKey()
            #cv.imwrite('result_click_point.jpg', haystack_img)

        return points
    
    def draw_rectangles(self, haystack_img, rectangles):
        # these colors are actually BGR
        line_color = (0, 255, 0)
        line_type = cv2.LINE_4

        for (x, y, w, h) in rectangles:
            # determine the box positions
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # draw the box)
            cv2.rectangle(haystack_img, top_left, bottom_right, line_color, lineType=line_type)

        return haystack_img


class WindowCapture:
    def __init__(self,window_title):
        # find the handle for the window we want to capture
        self.image = None
        self.window_title = window_title
        self.update_handle()
  

    def update_handle(self):
        # Try to find the window and update the window handle
        windows = gw.getWindowsWithTitle(self.window_title)
        self.hwnd = windows[0] if windows else None

    def get_window_image(self):
        win = gw.getWindowsWithTitle(self.window_title)[0]  
        if win:
            if win.isMinimized:
                win.restore()

            x, y, width, height = win.left, win.top, win.width, win.height
            img = pyautogui.screenshot(region=(x+50, y/2, width-100, height-400))
            #img = IG.grab().resize(x+50, y/2, width-100, height-400)
            #img = pyautogui.screenshot(region=(x/2, y/2, width, height/2))

            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        else:
            return None
        
    def get_screen_position(self):
        if self.hwnd:
            # Return a tuple of x, y, width, and height
            return self.hwnd.left, self.hwnd.top, self.hwnd.width, self.hwnd.height
        else:
            print("Window not found!")
            return None
    

       
    
window_title = "Night T-Rex Game - Brave"  # Replace with the title of your window

cascade_cactus  = cv2.CascadeClassifier(r'C:\Users\Alt21\Documents\Python_Projects\OpenCV_Test_2_Real_time\cascade_over\cascade.xml')
cascade_Trex = cv2.CascadeClassifier(r'C:\Users\Alt21\Documents\Python_Projects\OpenCV_Test_2_Real_time\cascade_T-rex\cascade.xml')

Vision_Trex = Vision(None)


window_capture = WindowCapture(window_title)
print(window_capture.get_screen_position())
   
JUMP_THRESHOLD = 250  
while True:
 
    start_time = time.time()  # Start time for calculating FPS
    img = window_capture.get_window_image()

 
   
    if img is not None:

        # Detect T-rex and cacti in the frame
        rectangles_TRex = cascade_Trex.detectMultiScale(img)
        rectangles_cactus = cascade_cactus.detectMultiScale(img)

        # Draw rectangles around T-rex and cacti
        Vision_Trex.draw_rectangles(img, rectangles_TRex)
        Vision_Trex.draw_rectangles(img, rectangles_cactus)

        x_trex = None
        y_trex = None
        x_closest_cactus = None
        y_closest_cactus = None


        for (x, y, w, h) in rectangles_TRex:
            x_trex = x 
            y_trex = y  
            print("T-rex: ", x_trex)
            break              
            
        for (x, y, w, h) in rectangles_cactus:
            if x_trex is not None:
            
                if x > x_trex and (x_closest_cactus is None or x < x_closest_cactus):
                    x_closest_cactus = x
                    y_closest_cactus = y
                    print("x_closest_cactus: ", x_closest_cactus)

        if x_trex is not None and x_closest_cactus is not None:
            distance_to_cactus = x_closest_cactus - x_trex
            if distance_to_cactus <= JUMP_THRESHOLD:
                print("Jump!")  
                print(distance_to_cactus)
                pyautogui.press('space')
                
        cv2.line(img,(x_trex,y_trex),(x_closest_cactus,y_closest_cactus),(0,255,0), 2)

        cv2.imshow('image',img)
        
        key = cv2.waitKey(1)
        '''if key == ord('q'):  # Exit if 'q' is pressed
            break
        elif key == ord('f'):
            cv2.imwrite(os.path.join(positive_dir, '{}.jpg'.format(start_time)), img)
        elif key == ord('d'):
            cv2.imwrite(os.path.join(negative_dir, '{}.jpg'.format(start_time)), img)'''
    else:
        break

    # FPS calculation
    fps = 1.0 / (time.time() - start_time)
    print("FPS: {:.2f}".format(fps))

cv2.destroyAllWindows()


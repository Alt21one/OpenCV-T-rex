import bettercam
import pygetwindow as gw
import time
import numpy as np
import cv2
import keyboard


camera = bettercam.create(output_idx=0, output_color="BGR")

left, top = (1920 - 240) // 2, (1080 - 640) // 2
right, bottom = left + 840, top + 340
region = (left, top, right, bottom)

WindowCap = True

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


def capture_window_screenshot(window_title):
    try:
        # Find the window by title
        window = gw.getWindowsWithTitle(window_title)[0]
        if window is not None:
            # If the window is minimized, restore it
            if window.isMinimized:
                window.restore()
            
            # Bring the window to the front (optional, for active window capture)
            #window.activate()
            # Get window bounds
            #left, top, width, height = window.left, window.top, window.width, window.height

            #print(left,"",top,"",width,"",height)
            # Capture the screenshot
            #frame = camera.grab(region=(left // 2, top // 2, width, height))
            frame = camera.grab(region=region)
            return frame
        else:
            print("Window not found.")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

cascade_Trex = cv2.CascadeClassifier(r'C:\Users\Alt21\Documents\Python_Projects\OpenCV_Test_2_Real_time\cascade_T-rex\cascade.xml')
cascade_cactus  = cv2.CascadeClassifier(r'C:\Users\Alt21\Documents\Python_Projects\OpenCV_Test_2_Real_time\cascade\cascade.xml')
Vision_Trex = Vision(None)


JUMP_THRESHOLD = 225
while WindowCap:
    start_time = time.time()
    frame = capture_window_screenshot('Night T-Rex Game - Brave')

    if frame is None:
        continue  # Skip this iteration

    rectangles_TRex = cascade_Trex.detectMultiScale(frame)
    rectangles_cactus = cascade_cactus.detectMultiScale(frame)

    # Draw rectangles around T-rex and cacti
    Vision_Trex.draw_rectangles(frame, rectangles_TRex)
    Vision_Trex.draw_rectangles(frame, rectangles_cactus)

    
    x_trex = None
    y_trex = None
    x_closest_cactus = None
    y_closest_cactus = None


    for (x, y, w, h) in rectangles_TRex:
            x_trex = x 
            y_trex = y  
            break              
            
    for (x, y, w, h) in rectangles_cactus:
            if x_trex is not None:
            
                if x > x_trex and (x_closest_cactus is None or x < x_closest_cactus):
                    x_closest_cactus = x
                    y_closest_cactus = y

    if x_trex is not None and x_closest_cactus is not None:
            distance_to_cactus = x_closest_cactus - x_trex
            if distance_to_cactus <= JUMP_THRESHOLD:
                keyboard.press('space')


                 
    #cv2.line(frame,(x_trex,y_trex),(x_closest_cactus,y_closest_cactus),(0,255,0), 2)

    cv2.imshow("Camera Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    fps = 1.0 / (time.time() - start_time)
    print("FPS: {:.2f}".format(fps))

camera.release()    
cv2.destroyAllWindows()



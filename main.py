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

    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None
    
    def __init__(self, needle_img_path, method=cv2.TM_CCOEFF_NORMED):
        if needle_img_path:
            self.needle_img = cv2.imread(needle_img_path, cv2.IMREAD_UNCHANGED)

            # Save the dimensions of the needle image
            self.needle_w = self.needle_img.shape[1]
            self.needle_h = self.needle_img.shape[0]

        self.method = method

    def find(self, haystack_img, threshold=0.5, debug_mode=None):

        if self.needle_img.shape[2] == 4:
            needle_mask = self.needle_img[:, :, 3]
            self.needle_img = self.needle_img[:, :, :3]
            result = cv2.matchTemplate(haystack_img, self.needle_img, self.method, mask=needle_mask)
        else:
            result = cv2.matchTemplate(haystack_img, self.needle_img, self.method)

        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]

            rectangles.append(rect)
            rectangles.append(rect)
        rectangles, weights = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

        points = []
        if len(rectangles):

            line_color = (0, 255, 0)
            line_type = cv2.LINE_4
            marker_color = (255, 0, 255)
            marker_type = cv2.MARKER_CROSS

            for (x, y, w, h) in rectangles:

                center_x = x + int(w/2)
                center_y = y + int(h/2)
                # Save the points
                points.append((center_x, center_y))

                if debug_mode == 'rectangles':

                    top_left = (x, y)
                    bottom_right = (x + w, y + h)

                    cv2.rectangle(haystack_img, top_left, bottom_right, color=line_color, 
                                lineType=line_type, thickness=2)
                elif debug_mode == 'points':

                    cv2.drawMarker(haystack_img, (center_x, center_y), 
                                color=marker_color, markerType=marker_type, 
                                markerSize=40, thickness=2)

        if debug_mode:
            cv2.imshow('Matches', haystack_img)


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

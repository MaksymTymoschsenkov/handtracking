import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import keyboard  # To handle global hotkeys
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size for cursor control
screen_width, screen_height = pyautogui.size()

tracking = False  # Initialize tracking as False
tapping_state = False  # To track the tapping state

def toggle_tracking(e=None):
    global tracking
    tracking = not tracking
    print(f"Tracking {'started' if tracking else 'stopped'}")

# Register global hotkey (Alt + Q) to toggle tracking
keyboard.add_hotkey('alt+q', toggle_tracking)

print("Press 'Alt + Q' to toggle hand tracking.")

# Helper function to calculate the distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)

# Helper function to check if a hand is in a tapping position
def is_tapping(index_tip, thumb_tip, img_width, img_height):
    index_x, index_y = int(index_tip.x * img_width), int(index_tip.y * img_height)
    thumb_x, thumb_y = int(thumb_tip.x * img_width), int(thumb_tip.y * img_height)
    distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)
    return distance < 40  # Adjust the threshold as needed

# Main loop
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Flip the image horizontally for a natural selfie-view display
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(img_rgb)
    
    if tracking and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates of index finger tip (landmark 8) and thumb tip (landmark 4)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            h, w, c = img.shape
            
            # Check if the index and thumb are in a tapping position
            if is_tapping(index_finger_tip, thumb_tip, w, h):
                if not tapping_state:
                    pyautogui.click()
                    tapping_state = True
                    time.sleep(0.2)  # Add a small delay to avoid multiple clicks
            else:
                tapping_state = False

            # Get the coordinates of the index finger tip for cursor movement
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Convert hand coordinates to screen coordinates
            screen_x = np.interp(index_x, [0, w], [0, screen_width])
            screen_y = np.interp(index_y, [0, h], [0, screen_height])
            
            # Move the mouse cursor
            pyautogui.moveTo(screen_x, screen_y)
            
            # Draw a circle at the index finger tip
            cv2.circle(img, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)

    # Display the image
    cv2.imshow("Hand Tracking", img)
    
    # Exit if 't' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
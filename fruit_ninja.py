import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random
import math
import urllib.request
import os

class Fruit:
    """Represents a flying fruit on the screen."""
    def __init__(self, screen_w, screen_h):
        self.radius = random.randint(30, 60)
        # Start somewhere at the bottom of the screen
        self.x = random.randint(self.radius, screen_w - self.radius)
        self.y = screen_h + self.radius
        
        # Initial upward velocity (negative y direction)
        self.vy = -random.uniform(18, 28)
        # Small horizontal drift
        self.vx = random.uniform(-4, 4)
        
        # Vibrant random colors for the fruits
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.active = True

    def update(self, gravity):
        """Update fruit position based on physics."""
        self.x += self.vx
        self.y += self.vy
        self.vy += gravity  # Apply gravity to vertical velocity

class Explosion:
    """Represents a visual explosion effect when a fruit is sliced."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.radius = 10
        self.alpha = 255  # Opacity, ranges from 0 to 255
        self.active = True

    def update(self):
        """Expand the radius and fade out."""
        self.radius += 8
        self.alpha -= 25
        if self.alpha <= 0:
            self.active = False

def download_model():
    """Downloads the required MediaPipe model file if it doesn't exist."""
    model_path = 'hand_landmarker.task'
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Downloading MediaPipe model ({model_path}) from {url}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    return model_path

def main():
    # 1. Download model file (Required for the newer MediaPipe Tasks API)
    model_path = download_model()

    # 2. Initialize Webcam
    # Using 0 for the default camera
    cap = cv2.VideoCapture(0)
    
    # Requesting specific resolution (1280x720)
    w, h = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # 3. Initialize MediaPipe Tasks API (HandLandmarker)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Game State Variables
    fruits = []
    explosions = []
    score = 0
    gravity = 0.6
    spawn_timer = 0

    print("Game Started. Press 'ESC' to exit.")

    # 4. Main Game Loop
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from webcam. Exiting...")
            break

        # Flip the frame horizontally for an intuitive mirror effect
        frame = cv2.flip(frame, 1)
        
        # In case the camera doesn't natively support 1280x720, force resize
        frame = cv2.resize(frame, (w, h))

        # MediaPipe needs RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame using the Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        index_tips = []

        # 5. Hand Tracking and Rendering
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Extract Landmark 8: Index Finger Tip
                index_finger = hand_landmarks[8]
                
                # Convert normalized coordinates to pixel coordinates
                cx = int(index_finger.x * w)
                cy = int(index_finger.y * h)
                index_tips.append((cx, cy))
                
                # Draw the "blade" (a glowing circle effect)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 255), -1)      # Inner core
                cv2.circle(frame, (cx, cy), 20, (0, 200, 255), 2)       # Outer ring
                cv2.circle(frame, (cx, cy), 25, (0, 150, 255), 1)       # Faint glow

        # 6. Game Physics: Spawn Fruits
        spawn_timer += 1
        # Randomly spawn fruits every ~20 frames
        if spawn_timer > 20:
            if random.random() > 0.4:  # 60% chance to spawn
                fruits.append(Fruit(w, h))
            spawn_timer = 0

        # Update & Draw Fruits
        for fruit in fruits:
            fruit.update(gravity)
            
            # If the fruit falls entirely off the bottom of the screen, mark it inactive
            if fruit.y > h + fruit.radius * 2:
                fruit.active = False
            else:
                # Draw fruit core and a white outline
                cv2.circle(frame, (int(fruit.x), int(fruit.y)), fruit.radius, fruit.color, -1)
                cv2.circle(frame, (int(fruit.x), int(fruit.y)), fruit.radius, (255, 255, 255), 3)

        # 7. Collision Detection (The Slice)
        for fruit in fruits:
            if not fruit.active:
                continue
            
            for cx, cy in index_tips:
                # Euclidean distance between Index Tip and Fruit Center
                dist = math.hypot(cx - fruit.x, cy - fruit.y)
                
                # If distance is less than the radius, we have a collision
                if dist < fruit.radius:
                    fruit.active = False
                    score += 10
                    # Trigger explosion effect at the fruit's position
                    explosions.append(Explosion(int(fruit.x), int(fruit.y), fruit.color))
                    break # Break inner loop since this fruit is already sliced

        # Remove inactive fruits from the list to save memory
        fruits = [f for f in fruits if f.active]

        # 8. Update & Draw Explosions (Slice Effects)
        for exp in explosions:
            exp.update()
            if exp.active:
                # To simulate a fading flash, we use cv2.addWeighted for transparency
                overlay = frame.copy()
                cv2.circle(overlay, (exp.x, exp.y), exp.radius, exp.color, -1)
                # Apply transparency based on the explosion's alpha value
                alpha_factor = max(0, exp.alpha / 255.0)
                cv2.addWeighted(overlay, alpha_factor, frame, 1 - alpha_factor, 0, frame)

        # Remove inactive explosions
        explosions = [e for e in explosions if e.active]

        # 9. Render UI
        # Background shadow for text readability
        cv2.putText(frame, f"Score: {score}", (22, 62), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 5)
        # Foreground text
        cv2.putText(frame, f"Score: {score}", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)

        # Show the frame
        cv2.imshow("Fruit Ninja AI", frame)

        # Wait for 1 millisecond; check if 'ESC' (ASCII 27) is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

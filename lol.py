import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        # Read image or video frame
        image = cv2.imread('hands3.jpg', cv2.IMREAD_UNCHANGED)


        # Process image with Mediapipe to detect hand landmarks
        results = hands.process(image)

        # Draw hand landmarks on image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Extract hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]

        # Identify points for fingertips and palm
        fingertips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                      hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                      hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                      hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                      hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]]
        palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        # Calculate rectangle width and height
        width = int(abs(fingertips[0].x - fingertips[3].x) * image.shape[1])
        height = int(abs(palm.y - max([tip.y for tip in fingertips])) * image.shape[0])

        # Draw rectangle on image
        cv2.rectangle(annotated_image, (int(fingertips[0].x * image.shape[1]), int(max([tip.y for tip in fingertips]) * image.shape[0])),
                      (int(fingertips[3].x * image.shape[1]), int(palm.y * image.shape[0])), (0, 255, 0), 2)

        # Display image with rectangle
        cv2.imshow('Hand', annotated_image)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

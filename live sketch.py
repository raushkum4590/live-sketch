import cv2

def sketch(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Detect edges in the blurred frame using the Canny edge detector
    edges = cv2.Canny(blurred_frame, 30, 100)
    
    # Invert the edges to obtain a negative image
    inverted_edges = 255 - edges
    
    # Combine the inverted edges with the grayscale frame using the "dodge" blend mode
    sketch_frame = cv2.divide(gray_frame, inverted_edges, scale=256.0)
    
    return sketch_frame

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam feed
    ret, frame = cap.read()
    if not ret:
        break
    
    # Generate the sketch from the frame
    sketch_frame = sketch(frame)
    
    # Display the original frame and the sketch frame
    cv2.imshow('Live Sketch', sketch_frame)
    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

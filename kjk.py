import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# Preprocessing parameters
CARD_WIDTH = 200
CARD_HEIGHT = 300

# Function to preprocess the card image
def preprocess_card(card_image):
    # Resize the card image to a fixed size
    resized_card = cv2.resize(card_image, (CARD_WIDTH, CARD_HEIGHT))
    
    # Convert the resized card image to grayscale
    gray_card = cv2.cvtColor(resized_card, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to enhance contrast
    equalized_card = cv2.equalizeHist(gray_card)
    
    # Extract HOG features from the preprocessed card image
    hog_features = hog(equalized_card, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    
    return hog_features

# Function to extract features from the dataset
def extract_features(dataset_path):
    features = []
    labels = []
    
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read the card image
            card_image = cv2.imread(os.path.join(dataset_path, filename))
            
            # Preprocess the card image
            feature_vector = preprocess_card(card_image)
            
            # Extract the label from the filename
            label = filename.split('_')[0]  # Assuming the filename format is "color_number.jpg"
            
            features.append(feature_vector)
            labels.append(label)
    
    return features, labels

# Function to train the classification model
def train_model(features, labels):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Create an SVM classifier with RBF kernel
    svm = SVC(kernel='rbf', C=10, gamma=0.1)
    
    # Train the classifier
    svm.fit(X_train, y_train)
    
    # Evaluate the classifier on the testing set
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    
    return svm

# Function to recognize the card
def recognize_card(card_image, model):
    # Preprocess the card image
    feature_vector = preprocess_card(card_image)
    
    # Reshape the feature vector to a 2D array
    feature_vector = feature_vector.reshape(1, -1)
    
    # Predict the card label using the trained model
    predicted_label = model.predict(feature_vector)
    
    return predicted_label[0]

# Function to detect and track cards in a frame
def detect_and_track_cards(frame, model):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Perform edge detection using Canny algorithm
    edges = cv2.Canny(blurred_frame, 50, 150)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    recognized_cards = []
    
    # Iterate over the contours
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter contours based on size and aspect ratio
        aspect_ratio = w / float(h)
        if w > 50 and h > 50 and 0.7 < aspect_ratio < 1.3:
            # Extract the card region from the frame
            card_image = frame[y:y+h, x:x+w]
            
            # Recognize the card
            predicted_label = recognize_card(card_image, model)
            
            recognized_cards.append((predicted_label, (x, y, w, h)))
    
    return recognized_cards

# Main function
def main():
    # Path to the dataset directory
    dataset_path = 'C:\\Users\\MAHIM TRIVEDI\\Downloads\\UNO'
    
    # Extract features from the dataset
    features, labels = extract_features(dataset_path)
    
    # Train the classification model
    model = train_model(features, labels)
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Detect and track cards in the frame
        recognized_cards = detect_and_track_cards(frame, model)
        
        # Draw bounding boxes and labels for recognized cards
        for card_label, (x, y, w, h) in recognized_cards:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, card_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('UNO Card Recognition', frame)
        
        # Check for 'q' key to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == '__main__':
    main()
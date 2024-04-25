UNO Card Recognition System
Overview
This project develops a system to recognize UNO cards from images or a live camera feed using Python, OpenCV, and a Support Vector Machine (SVM) classifier. The system preprocesses images to extract Histogram of Oriented Gradients (HOG) features, which are then used to train an SVM for recognizing the card type based on its color and number.

Requirements
Python 3.8 or higher
OpenCV
NumPy
scikit-learn
scikit-image
A webcam for live detection (optional)
Installation
Python Installation: Ensure Python 3.8 or higher is installed on your system. If not, download and install it from python.org.
Dependency Installation: Install the required Python libraries using pip:
bash
Copy code
pip install numpy opencv-python scikit-learn scikit-image
Dataset
The dataset should consist of images of UNO cards, each labeled according to the card's color and number. Store your dataset in a structured directory as follows:

makefile
Copy code
C:\Users\MAHIM TRIVEDI\Downloads\UNO
Each image should be named in the format color_number.jpg (e.g., red_3.jpg).

Usage
To run the card recognition system, execute the main script. This can be done from the command line as follows:

bash
Copy code
python uno_card_recognition.py
This will start the webcam and begin detecting and recognizing UNO cards in real time. To quit the application, press 'q' while the focus is on the video window.

Features
Image Preprocessing: Converts images to grayscale, applies histogram equalization to enhance contrast, and extracts HOG features.
Card Detection: Uses edge detection and contour finding to locate cards within the video frame.
Card Recognition: Classifies the detected cards using a pre-trained SVM model based on HOG features.
Functions Description
preprocess_card(card_image): Processes an image of a card and extracts HOG features.
extract_features(dataset_path): Loops through the dataset, preprocesses each image, and extracts features and labels.
train_model(features, labels): Trains an SVM classifier using the provided features and labels.
recognize_card(card_image, model): Recognizes an individual card from its image using the trained model.
detect_and_track_cards(frame, model): Detects and recognizes cards from a video frame.
main(): Main function to execute the program, handling dataset loading, model training, and real-time card recognition.
Contributing
Feel free to fork this repository and submit pull requests to enhance the functionalities of the UNO card recognition system. You can also open issues for any bugs found or improvements suggested.

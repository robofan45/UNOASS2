
# UNO Card Recognition System

This project aims to develop a computer vision system for recognizing and tracking UNO cards from images or video streams using machine learning techniques. The system utilizes Python, OpenCV for image processing, and Scikit-learn for implementing a Support Vector Machine (SVM) classifier.

## Features

- Real-time card recognition from a webcam feed
- Ability to process static images
- Histogram of Oriented Gradients (HOG) for feature extraction
- SVM classifier for card recognition
- Data augmentation and hyperparameter tuning for improved accuracy
- Customizable for different datasets or classification tasks

## Prerequisites

Before running the project, ensure you have the following prerequisites installed:

- Python 3.8+
- OpenCV
- NumPy
- scikit-learn
- scikit-image

You can install the required libraries using the following command:

```
pip install numpy opencv-python scikit-learn scikit-image
```

## Dataset Structure

Organize your dataset in the following manner within your project directory:

```
/path/to/your/dataset
├── red_1.jpg
├── red_2.jpg
├── green_1.jpg
├── yellow_skip.jpg
├── ...
```

Each image file name should represent the card's color and label.

## Usage

1. Clone the repository:

```
git clone https://your-repository-link.git
cd uno_card_recognition
```

2. Update the `dataset_path` in the `main()` function to point to your dataset directory.

3. Run the script:

```
python uno_card_recognition.py
```

4. For real-time card recognition, place UNO cards in front of the webcam.
5. To quit the live camera feed, press the 'q' key.

## System Operation

1. **Start**: Launch the system using the command above.
2. **Operation**: Place UNO cards in front of the camera or provide images through the file input system.
3. **Quit**: Press 'q' to quit the live camera feed.

## Contributing

Contributions to the UNO Card Recognition System are welcome. Please fork the repository and submit pull requests with your enhancements. For bugs or suggestions, open an issue in the repository.

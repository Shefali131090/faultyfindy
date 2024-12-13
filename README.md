Project Summary: Tyre Defect Detection using Deep Learning
Objective:
The goal of this project is to develop a machine learning model that can classify images of tyres into two categories:

Good Tyres
Defective Tyres
The model is built using Convolutional Neural Networks (CNN) to automatically detect defects in tyres based on images.

Steps Taken:
Dataset Preparation:

The dataset consists of images of tyres categorized into good and defective tyres.
The images were organized in two folders: good and defective.
Data Preprocessing:

Image Loading: Images were loaded from their respective folders and resized to a consistent shape for input to the CNN model.
Normalization: Image pixel values were normalized to a range of 0 to 1 by dividing the pixel values by 255.
Data Splitting: The dataset was split into training and testing sets using an 80-20 split.
Image Augmentation: Applied data augmentation techniques such as rotation, zoom, and flips to enhance the dataset and improve model generalization.
Model Architecture:

A Convolutional Neural Network (CNN) model was defined with the following layers:
Convolutional Layers: Extract features from the images.
Max-Pooling Layers: Reduce dimensionality and computational complexity.
Flatten Layer: Convert 2D feature maps into a 1D vector.
Fully Connected Layers: Perform classification based on the extracted features.
Dropout Layer: Prevent overfitting by randomly dropping units during training.
Model Training:

The model was trained for 10 epochs with a batch size of 32.
The training process included tracking accuracy and loss metrics to monitor the model's performance.
During training, the model achieved an accuracy of about 65% on the training data.
Model Evaluation:

After training, the model was evaluated on the test dataset.
The model achieved an accuracy of 55.6% on the test set, but the precision and recall for the "defective" class were low due to class imbalance.
The model's performance metrics were:
Precision: 0.56 for "good" tyres, and 0 for "defective" tyres.
Recall: 1.0 for "good" tyres, and 0.0 for "defective" tyres.
F1-Score: 0.71 for "good" tyres, and 0.0 for "defective" tyres.
Challenges:

The model faced significant challenges due to the class imbalance (many more "good" tyres than "defective" tyres), which resulted in biased predictions towards the "good" class.
This led to the low performance in detecting "defective" tyres, as seen in the poor precision and recall for that class.
Next Steps:

Class Imbalance Handling: Implement techniques like class weighting, oversampling the minority class, or undersampling the majority class to address the imbalance.
Hyperparameter Tuning: Tune model parameters such as the learning rate, batch size, or number of layers to improve performance.
Model Improvement: Explore more advanced models, such as pre-trained CNNs (e.g., VGG16, ResNet), or fine-tune the existing architecture.
Threshold Adjustment: Adjust the decision threshold to improve recall for the "defective" class.
Model Deployment: Once satisfied with the model, deploy it using a web framework or cloud platform for real-time tyre defect detection.
Conclusion:
The model is a starting point for automating the classification of tyre defects. However, the performance is impacted by class imbalance, which can be mitigated through further improvements like data balancing, tuning, and model enhancements.

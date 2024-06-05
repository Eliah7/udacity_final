# High-level overview
In this project, I developed a dog breed classifier capable of identifying the breed of a dog from an image. The problem I aimed to solve is significant because accurate dog breed classification can have various applications, such as in veterinary care, pet adoption, and lost pet recovery. By automating this process with a robust machine learning model, I aim to provide an efficient and reliable solution for identifying dog breeds.

# Metrics
The metric that was used in this project is the accuracy metric.
## Accuracy
This metric refers to how close a result or prediction is to the true or correct value. It is a measure of correctness or the level of agreement between a model's output and the actual expected outcome.

Accuracy is a commonly used metric to evaluate the performance of Convolutional Neural Networks (CNNs), especially in classification tasks. Here's a detailed explanation of the accuracy metric in this context:

##Definition
Accuracy is the ratio of the number of correct predictions to the total number of predictions. It is calculated as:

```
Accuracy = Number of Correct Predictions/Total Number of Predictions
```

# Description of Input Data
The dataset used for this project is the Stanford Dogs Dataset, which contains over 6675 training images of 133 different dog breeds. The dataset is well-structured, with images labeled by breed. This labeling is crucial for training a supervised learning model. The images vary in size and quality, providing a realistic and challenging dataset for classification tasks.

### Variables in the dataset:

- Images: The primary data consisting of photographs of dogs.
- Labels: The breed of each dog, which serves as the target variable for our classification model.

# EDA
The Image Dimensions Distribution visualization helps in understanding the variability in the widths and heights of images in the dataset. By plotting histograms or box plots of image dimensions, one can detect whether there is a significant variation in image sizes. This information is crucial for preprocessing steps, such as resizing images to a consistent size before feeding them into the CNN. It also helps in identifying if any images have unusual dimensions that could potentially cause issues during training. Overall, analyzing image dimensions ensures that the data is uniformly prepared for optimal model performance. In this model all images were resized to 224 x 224.

![alt text](https://github.com/Eliah7/udacity_final/blob/master/image_distribution.png)


# Strategy for solving the problem
The overall approach to solving the problem involved several key steps:

1. Data Exploration: Understanding the dataset through exploratory data analysis (EDA).
2. Data Preprocessing: Cleaning and preparing the data for model training.
3. Model Development: Building a Convolutional Neural Network (CNN) from scratch, followed by using transfer learning with pre-trained models.
4. Evaluation: Assessing the model's performance using appropriate metrics.

# Discussion of the expected solution
The proposed solution involves a multi-step process. Initially, I built a CNN from scratch to classify dog breeds. This provided a baseline accuracy. Given the complexity of the task, I then employed transfer learning using pre-trained models like VGG19 and ResNet50 to leverage their feature extraction capabilities. The overall workflow includes image preprocessing, feature extraction, and classification.


# Data Preprocessing
Data preprocessing involved several steps:

1. Resizing Images: Ensuring all images are of a consistent size (224x224 pixels) for model input.
2. Normalization: Scaling pixel values to a range of 0 to 1 to improve model convergence.

# Modeling
Initially, a CNN was built from scratch using Keras. However, to achieve higher accuracy, I employed transfer learning with pre-trained models such as VGG19 and ResNet50. These models have been pre-trained on large datasets and can extract features effectively.

# Results
The final model, using transfer learning, achieved a significantly higher accuracy compared to the CNN built from scratch. Key results included:

Accuracy: 83% on the test set. Below is a comparison of the models that were developed in the project.

| Model         | Test Accuracy |
|---------------|---------------|
| CNN From Scratch  |  1.3822%  |
| VGG16  |  0.8383%           |
| VGG19   |  0.7186%           |



# Deployment
The final model was deployed using the flask framework to provide an interface where a user can upload an image and the model return the classification of the model.

To run the model do the following:
1. Navigate to the app directory
` cd app`
2. Make the setup.sh executable and execute it
```
chmod +x setup.sh
./setup.sh
```
3. Install all the required libraries
`pip install -r requirements.sh`
4. Run the app and use it http:localhost:5000
`python run.py`

# Conclusion 
This project successfully developed a dog breed classifier with high accuracy using transfer learning. The model can be applied in various domains where identifying dog breeds quickly and accurately is essential. Our findings demonstrate the effectiveness of transfer learning in image classification tasks, particularly with diverse and complex datasets like dog breeds.

## Reflection
Experimenting with different CNN architectures was another intriguing aspect. Designing and tuning the layers, from convolutional layers to pooling and fully connected layers, offered a hands-on understanding of how each component contributes to the model's ability to learn from image data.

Using other models weights to train a custom CNN model through transfer learning was an eye opening solution that helps to deal with the challenge of poor results.

# Improvements
Future improvements could include:

1. Incorporating more data: Expanding the dataset to include more images for underrepresented breeds.
2. Advanced augmentation techniques: Using more sophisticated data augmentation to further improve model robustness.
3. Model ensemble: Combining predictions from multiple models to improve accuracy and generalization.

# Acknowledgment
I thank Udacity and Vodafone for making it possible for me to do this project.


# High-level overview
In this project, we developed a dog breed classifier capable of identifying the breed of a dog from an image. The problem we aimed to solve is significant because accurate dog breed classification can have various applications, such as in veterinary care, pet adoption, and lost pet recovery. By automating this process with a robust machine learning model, we aim to provide an efficient and reliable solution for identifying dog breeds.

# Description of Input Data
The dataset used for this project is the Stanford Dogs Dataset, which contains over 6675 training images of 133 different dog breeds. The dataset is well-structured, with images labeled by breed. This labeling is crucial for training a supervised learning model. The images vary in size and quality, providing a realistic and challenging dataset for classification tasks.

### Variables in the dataset:

- Images: The primary data consisting of photographs of dogs.
- Labels: The breed of each dog, which serves as the target variable for our classification model.

# Strategy for solving the problem
The overall approach to solving the problem involved several key steps:

1. Data Exploration: Understanding the dataset through exploratory data analysis (EDA).
2. Data Preprocessing: Cleaning and preparing the data for model training.
3. Model Development: Building a Convolutional Neural Network (CNN) from scratch, followed by using transfer learning with pre-trained models.
4. Evaluation: Assessing the model's performance using appropriate metrics.

# Discussion of the expected solution
The proposed solution involves a multi-step process. Initially, we built a CNN from scratch to classify dog breeds. This provided a baseline accuracy. Given the complexity of the task, we then employed transfer learning using pre-trained models like VGG19 and ResNet50 to leverage their feature extraction capabilities. The overall workflow includes image preprocessing, feature extraction, and classification.


# Data Preprocessing
Data preprocessing involved several steps:

1. Resizing Images: Ensuring all images are of a consistent size (224x224 pixels) for model input.
2. Normalization: Scaling pixel values to a range of 0 to 1 to improve model convergence.

# Modeling
Initially, a CNN was built from scratch using Keras. However, to achieve higher accuracy, we employed transfer learning with pre-trained models such as VGG19 and ResNet50. These models have been pre-trained on large datasets and can extract features effectively.

# Results
The final model, using transfer learning, achieved a significantly higher accuracy compared to the CNN built from scratch. Key results included:

Accuracy: 85% on the test set.

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



# Improvements
Future improvements could include:

1. Incorporating more data: Expanding the dataset to include more images for underrepresented breeds.
2. Advanced augmentation techniques: Using more sophisticated data augmentation to further improve model robustness.
3. Model ensemble: Combining predictions from multiple models to improve accuracy and generalization.

# Acknowledgment
I thank Udacity and Vodafone for making it possible for me to do this project.


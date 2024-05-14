# Artistic Image Aesthetic Assessment

This repository contains the implementation of our convolutional neural network (CNN) model for the automatic aesthetic assessment of artistic images. Unlike traditional models that primarily rely on large-scale photography datasets, our approach focuses on the unique challenges posed by artistic images, which are more complex, diverse, and abstract.

## Overview

The aesthetic assessment of images is a popular research topic due to its practical applications in various fields such as image recommendation, image ranking, and image search. Most current research on image aesthetic assessment relies on datasets composed primarily of user-taken photos in real-world scenarios, such as AVA and AADB. However, few studies have specifically addressed the automatic aesthetic assessment of artistic images.

In this project, we propose a convolutional neural network model that automatically generates aesthetic scores for input artistic images. Our approach distinguishes itself by incorporating artistic theories and analyzing aesthetic features in artistic images from three dimensions: color, brightness, and contour. These features are integrated to produce an overall aesthetic score.

## Dataset

We use a large-scale dataset of artistic images for aesthetic assessment, which consists of over 7,000 artistic images. Each image in the dataset is accompanied by an average aesthetic score assigned by users.

## Key Features

- **Three-Dimensional Feature Analysis**: Our model analyzes aesthetic features from three key dimensions:
  - **Color**
  - **Brightness**
  - **Contour**

## Model

The convolutional neural network (CNN) architecture is designed to handle the complexities and abstractions of artistic images. By integrating the three-dimensional aesthetic features, our model provides a robust assessment of artistic image aesthetics.

## Results

![Example Image](performance.png)
The comparison results between the predicted scores by our network and the true user ratings for all images in the testing set. The predicted scores closely align with the true user ratings.

## Getting Started

To get started with our model, follow the instructions below:

1. Train the network:
    ```bash
    python train.py
    ```
2. Evaluate a test image:
    ```bash
    python train.py --generate True
    ```

## Contributing

We welcome contributions to enhance the capabilities and performance of our aesthetic assessment model. Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or issues, please open an issue on GitHub or contact us at yansimin@zju.edu.cn


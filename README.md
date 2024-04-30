# Array of Sunshine

## Introduction
An AI program capable of detecting articles of clothing from images. This program will be particularly useful for platforms like eBay or any e-commerce website, where users sell a wide range of products. Categorization is essential for search and recommendation systems.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributors](#contributors)
- [License](#license)

## Installation
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage
To use this application:
1. Ensure Python and the required packages are installed.
2. Run the script using:
   ```bash
   python main.py
   ```

## Features
- **Data Preprocessing:** Automatically scales image data.
- **Model Training:** Trains a neural network on the Fashion MNIST dataset.
- **Evaluation:** Evaluates the model's performance using accuracy metrics.
- **Visualization:** Displays predictions for randomly selected images from the test dataset.

## Dependencies
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

## Documentation
The project is documented through comments within the code, explaining major functionalities and the flow of the application.

## Examples
After running the script, the model will output the accuracy of the test dataset and show a randomly chosen fashion item, its predicted and actual class labels.

## Contributors
To contribute to this project, please fork the repository and submit a pull request.

## License

This project is dual-licensed under both the MIT License and the Apache License, Version 2.0. You may choose to use either license depending on the requirements of your project.

### MIT License

The MIT License is a permissive license that allows users considerable freedom in using the software. It requires only that the copyright notice and the license text are included in any copies of the software. For the full text of the MIT License, see [MIT License](https://opensource.org/licenses/MIT).

### Apache License, Version 2.0

The Apache License, Version 2.0, also a permissive license, includes provisions on patent rights, which the MIT License does not cover. This license provides an express grant of patent rights from contributors to users. For the full text of the Apache License, Version 2.0, see [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

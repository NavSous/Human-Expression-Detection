# PyTorch Human Expression Detection

This repository contains the image files and PyTorch code to train a computer vision model to detect human expressions. 

All the required libraries for this project are available in requirments.txt

The images are from a kaggle database: https://www.kaggle.com/c/emotion-detection-from-facial-expressions

To train the model, adjust the settings for the training in main.py, and run the file. 

To test the model, go to the file test.py, and insert the path to any full color image into the correct location in the file:

```
im = load_image('./individual_test/your-file-name.extension')
```


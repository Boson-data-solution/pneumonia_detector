# Pneumonia Detection using X-Ray Images

This project aims to practice using CNN to process images and detect pneumonia based on the X-Ray images. Given the current COVID-19 pandemic, this project is both meaningful and interesting. Along with physical examination, imaging diagnosis plays a central role in the detection of pneumonia. In the chest X-Ray images, opacity areas are often correlated to pneumonia affected regions. However, the identification of opacity areas in chest X-Ray images is sometimes challenging. Machine learning and artificial intelligence can be used to detect pneumonia based on chest X-Ray images.

## 1. The dataset

The dataset for this project is an adapted version of dataset submitted by Paul Mooney at [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), which contains a more balanced train-validation-test split of the data. In total, there are 5856 observations in the dataset that is split into 4192 training examples (1082 normal and 3110 opacity), 1040 validation examples (267 normal and 773 opacity), and 624 testing examples (234 normal and 390 opacity).

All the chest X-ray images were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients routine clinical care.

# Sports Image Classification

### Project Summary:
This project creates a machine learning model to classify images into multiple sports categories using TensorFlow/Keras. The number of sports' categories is **73**.

---

### Dataset Description:

This dataset consists of `11146` images in **73 Sports classes**. Below are sample images from some of the classes present in the dataset:

![](./visuals/sports_classification_image.jpg?raw=true)

This train data was further split into a `80-20` train-validation split to perform training and evaluation.

---

### Machine Learning Model Architecture:

I used pretrained model `EfficientNetB0` with the `imagenet`'s pretrained weight and added **GlobalAveragePooling2D** layer and **Dropout** layer followed by the output layer to match with the project's requirement which is **73**.

---

### Data Augmentation:

For the training dataset, I applied the following data augmentation to avoid overfitting:

```
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomZoom(0.2),
  tf.keras.layers.RandomHeight(0.2),
  tf.keras.layers.RandomWidth(0.2),
], name="data_augmentation")
```

---

### Training Hyperparameters:

* Epochs: `100` with `EarlyStopping` with patience of `2`
  
* Optimizer: `adam`

* Learning Rate: `0.0001`

* LR scheduler: `lr_scheduler`

* Batch size: `32`

---

### Loss and Accuracy:

![](./visuals/sports_classification_loss.png?raw=true)

![](./visuals/sports_classification_accuracy.png?raw=true)

---

### Accuracy on Test Dataset for Kaggle Submission

The configurations discussed above, yielded a score of **0.91921** on the Kaggle's Leaderboard.

![](./visuals/sports_classification_kaggle_leaderboard.png?raw=true)

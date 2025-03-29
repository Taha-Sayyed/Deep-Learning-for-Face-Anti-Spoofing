# **TensorFlow Training Script Explanation**

This script trains a deep learning model using TensorFlow 1.x for binary classification (2 classes). It loads images, processes them into batches, and trains a model defined in `model_c.py`.

## **1. Import Required Libraries**
```python
import os
import numpy as np
import tensorflow as tf
import input_getbatch  # Custom module for loading and batching images
import model_c  # Model definition file
```
- `os` → Handles file paths and directories.
- `numpy` → Used for numerical computations.
- `tensorflow` → Deep learning framework used to build and train the model.
- `input_getbatch` → Custom module for loading images and creating batches.
- `model_c` → Defines the model structure and training operations.

---

## **2. Define Constants**
```python
N_CLASSES = 2  # Binary classification (2 classes)
IMG_W = 96  # Image width
IMG_H = 96  # Image height

BATCH_SIZE = 16  # Number of images per batch
CAPACITY = 4000  # Capacity of the input queue
MAX_STEP = 4000  # Maximum number of training steps
learning_rate = 0.0001  # Learning rate for optimization
```

---

## **3. Define the `run_training()` Function**
```python
def run_training():
```
This function performs the entire training process, including:
- Loading image data.
- Defining placeholders for input data.
- Building the model.
- Calculating loss and accuracy.
- Running the training loop.

---

## **4. Load Training Data**
```python
    train_dir = 'D:\\database\\train\\'  # Directory containing training images
    logs_train_dir = 'D:\\train_myself\\model_testc\\'  # Directory to save model checkpoints

    train, train_label = input_getbatch.get_files(train_dir)    
    train_batch, train_label_batch = input_getbatch.get_batch(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)
```

---

## **5. Define Input Placeholders**
```python
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])  
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])  
```

---

## **6. Build the Model**
```python
    logits = model_c.inference(x, BATCH_SIZE, N_CLASSES, 0.5)
    loss = model_c.losses(logits, y_)  
    acc = model_c.evaluation(logits, y_)
    train_op = model_c.trainning(loss, learning_rate)
```

---

## **7. Start TensorFlow Session**
```python
    with tf.Session() as sess:
        saver = tf.train.Saver()    
        sess.run(tf.global_variables_initializer())    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
```

---

## **8. Training Loop**
```python
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                
                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x: tra_images, y_: tra_labels})
```

---

## **9. Print Training Progress**
```python
                if step % 100 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
```

---

## **10. Save Model Checkpoints**
```python
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
```

---

## **11. Handle Errors and Stop Training**
```python
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)
```

---

## **Summary of Training Process**
1. **Load dataset** from `D:\database\train\`.
2. **Create batches** of 16 images (96×96).
3. **Define placeholders** for input data.
4. **Build the model** (`model_c.inference()`).
5. **Compute loss and accuracy**.
6. **Train using optimizer** (`model_c.trainning()`).
7. **Start a TensorFlow session**.
8. **Run training loop for 4000 steps**.
9. **Print loss & accuracy every 100 steps**.
10. **Save model checkpoints every 2000 steps**.
11. **Stop training and clean up resources**.

---

This markdown file provides a **detailed breakdown** of your TensorFlow script. You can now copy and paste it into your documentation. 🚀


## **Step-by-Step Explanation of the Code**

This script is used to preprocess and prepare image data for training a CNN model using TensorFlow. It consists of two functions:

1. **`get_files(file_dir)`** → Loads image file paths and assigns labels.  
2. **`get_batch(image, label, image_W, image_H, batch_size, capacity)`** → Creates batches of images and labels for training.  

---

## **1. `get_files(file_dir)`**
### **Purpose:**  
- Reads images from a specified directory (`file_dir`).
- Separates them into **real** and **fake** categories based on file names.
- Assigns labels:  
  - **Real images → Label 1**
  - **Fake images → Label 0**
- Shuffles the dataset to ensure randomness.

### **Step-by-step Breakdown:**
```python
def get_files(file_dir):
```
- Defines the function that takes `file_dir` as an input (folder path containing images).

```python
real = []
label_real = []
fake = []
label_fake = []
```
- Creates empty lists to store real and fake image file paths and their corresponding labels.

```python
for file in os.listdir(file_dir):
```
- Loops through all the files in `file_dir`.

```python
name = file.split(sep='.')
```
- Splits the filename by `.` to extract the first part (e.g., `real.1.png` → `['real', '1', 'png']`).

```python
if name[0]=='real':
    real.append(file_dir + file)  
    label_real.append(1)
```
- If the filename starts with `"real"`, it is added to the `real` list with label `1`.

```python
if name[0]=='fake':
    fake.append(file_dir + file)
    label_fake.append(0)
```
- If the filename starts with `"fake"`, it is added to the `fake` list with label `0`.

```python
print('There are %d real images\nThere are %d fake images' %(len(real), len(fake)))
```
- Prints the number of **real** and **fake** images found.

```python
image_list = np.hstack((real, fake))
label_list = np.hstack((label_real, label_fake))
```
- Combines real and fake images into a **single list** (`image_list`) and their labels into another list (`label_list`).

```python
temp = np.array([image_list, label_list])
temp = temp.transpose()
np.random.shuffle(temp)
```
- Creates a **2D NumPy array** where:
  - First row contains image file paths.
  - Second row contains corresponding labels.
- **Transposes** the array to align images and labels in pairs.
- **Shuffles** the array to randomize the dataset.

```python
all_image_list = temp[:, 0]  
all_label_list = temp[:, 1]   
```
- Extracts shuffled **image paths** and **labels** separately.

```python
all_label_list = [int(float(i)) for i in all_label_list]
```
- Converts label values from string/float to integers.

```python
return all_image_list, all_label_list
```
- Returns the shuffled **image paths** and **labels**.

---

## **2. `get_batch(image, label, image_W, image_H, batch_size, capacity)`**
### **Purpose:**
- Takes the shuffled image paths and labels from `get_files()`.
- Reads and processes image files.
- Converts them into batches for training.

### **Step-by-step Breakdown:**
```python
image = tf.cast(image, tf.string) 
label = tf.cast(label, tf.int32)
```
- Converts `image` paths to **TensorFlow string tensors**.
- Converts `label` values to **int32 tensors**.

```python
input_queue = tf.train.slice_input_producer([image, label])
```
- Creates a queue of images and labels for processing.
- Helps in parallel data loading for better efficiency.

```python
label = input_queue[1]
image_contents = tf.read_file(input_queue[0])
```
- Reads an image file from the queue.

```python
image = tf.image.decode_png(image_contents, channels=3)
```
- Decodes the **PNG** image (ensuring it has 3 color channels).

```python
image = tf.image.resize_images(image, (image_W, image_H), 0)
```
- Resizes the image to the specified width (`image_W`) and height (`image_H`).

```python
image = tf.image.per_image_standardization(image)
```
- Normalizes pixel values (subtracts mean and divides by standard deviation).

```python
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=batch_size,
                                          num_threads=64, 
                                          capacity=capacity)
```
- Groups images and labels into **mini-batches** for training.
- Uses **64 threads** for parallel processing.
- `capacity` ensures a buffer for loading images efficiently.

```python
label_batch = tf.reshape(label_batch, [batch_size])
image_batch = tf.cast(image_batch, tf.float32)
```
- Reshapes the labels into the correct format.
- Ensures the image batch is in **float32** format.

```python
return image_batch, label_batch
```
- Returns **image batches** and **label batches** for model training.

---

## **Summary**
1. **`get_files(file_dir)`**
   - Reads images from `file_dir`, assigns labels (1 for real, 0 for fake).
   - Shuffles and returns image paths and labels.

2. **`get_batch(image, label, image_W, image_H, batch_size, capacity)`**
   - Reads and processes images.
   - Converts them into mini-batches for training.

This code is used to **prepare data** for a deep learning model that detects real vs. fake faces in **face anti-spoofing tasks**. Let me know if you need any modifications! 🚀

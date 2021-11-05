# Deep learning (for audio) with Python

This document summarize what I have been learning on Deep learning for audio with Python, guided by Valerio Velardo

Most of the materials are extracted from his tutorials, and included some notes of my understanding on the subjects

## Takeaway points:

- AI = Building rational agents that act to achieve their goals their their beliefs
- ML is a subset of AI
- There are different flavours of ML and many ML algorithms
- DL is a subset of ML using DNNs
- DL isn't always the way to go
- Aritificial neurons are loosely inspired to biological neurons
- Artificial neurons are computational units
- They transform inputs into outputs using an activation function

## Understanding the audio data

### Fourier transform

- Decompose complex periodic sound into sum of sine waves oscillating at different frequencies

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled.png)

→ The amplitude tells us how much a specific frequency contribute to the complex sound

- Spectrum: Indicating the magnitude of a certain frequencies

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%201.png)

![Spectrum of this above piano sounds](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%202.png)

Spectrum of this above piano sounds

⇒ Moving from *time domain* to *frequency domain*

⇒ No time information

### Short time Fourier Transform (STFT)

- Compute several FFT at different intervals
    
    → Preserve  time information
    
- Fixed frame size (2048 samples) ⇒ Only FT those samples
- Gives a spectrogram (time + frequency + magnitude)

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%203.png)

→ Describe how much a frequency presents in the sound at a given time

FT used for this step is Fast Fourier Transform (FFT)

### Traditional ML pre-processing pipeline for audio data

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%204.png)

### Mel frequency Cepstral Coefficients (MFCCs)

- Capture timbral / textural aspects of the sound
- Frequency domain feature
- Approximate human auditory system
- 13 to 40 coefficients
- Calculated at each frame

![MFCC for the piano key sound](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%205.png)

MFCC for the piano key sound

- n_fft — window length for each time section
- hop_length — number of samples by which to slide the window at each step. Hence, the width of the Spectrogram is = Total number of samples / hop_length

## Rectified Linear Unit (ReLU)

- Better convergence
- Reduce likelihood of *vanishing gradient* (gradient value during epochs can be so small that it vanished - sigmoid function reduced the gradient by 0.25 per backward propagation)

## Type of batching

- Stochastic (batch_size = 1)
    - Calculate gradient on 1 sample
    - Quick, but inaccurate
- Full batch
    - Compute gradient on the whole training set
    - Slow, memory intensive, accurate
- Mini batch
    - Compute gradient on a subset of data set (16 - 128 samples)
    - Best of both worlds

## Overfitting:

Where the accuracy on the training set is perform better than the validation set

### Solving overfitting:

- Simpler architecture
    - Remove layers
    - Decrease # of neurons
    - ⇒ No universal rules
- Data Augmentation → More data you have, better model perform (Only for train dataset)
    - Artificially increase # of training samples
    - Apply transformations to audio files
        - Pitch shifting
        - Time stretching
        - Adding background noise
        - ...
- Early Stopping
    - Choose rule to stop training → Stop be fore overfitting
- Dropout
    - Randomly drop neurons while training
    - Increased network robustness
        - The network can't rely on some specific neurons too much
    - Dropout probability: 0.1-0.5
- Regularization
    - Add penalty to error function
    - Punish large weights
    - L1 and L2 regularization (L1 for simple to learn data, L2 for the complex one)
    
    ![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%206.png)
    
    ![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%207.png)
    

## Convolutional Neuron Network (CNN)

### CNNs

- Mainly used for processing images
- Perform better than multilayer perceptron
- Less parameters than dense layers

### Intuition

- Image data is structured
    - Edges, shapes
    - Translation invariance
    - Scale invariance (Square remains a square despite how big it is)
- CNN emulates human vision system
- Components of a CNN learn to extract different features

### CNN components

- Convolution
- Pooling

**Convolution:**

- Kernel (or filter) = grid of weights

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%208.png)

- Kernel is "applied" to the image
- Traditionally used in image processing
- Example:

![A sample representation of an image to a simple matrix](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%209.png)

A sample representation of an image to a simple matrix

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2010.png)

The output of the convolution is the same dimension matrix with the original image

Overlay the kernel on top of the image by:

$$\sum_{i=1}^{P}image_i\cdot K_{i}$$

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2011.png)

Then slide the kernel to the right

This way only the internal cells are calculated

⇒ Solve this by zero-padding

**Convolution: Zero padding**

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2012.png)

### Kernels

- Feature detector
- Kernels are learned
    - The network itself learned the kernel it needs to extract in order to perform well in the tasks
    - Learn the kernels meaning learn the value, the weight in the kernels

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2013.png)

### Architectural decisions for convolution

- Grid size
- Stride
- Depth
- Number of kernels

**Grid size:**

- # of pixels for height/width of the grid
- Odd numbers 
⇒ Having a central value → Use the center of the image as the start point to run the convolution

**Stride:**

- Step size used for sliding kernel on the image
- Indicated in pixels

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2014.png)

![Striding by 1 pixel](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2015.png)

Striding by 1 pixel

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2014.png)

![Striding by 2 pixels](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2016.png)

Striding by 2 pixels

**Depth:**

For grayscale image: Depth = 1

For color image (RGB for example): Depth = 3

Kernel = 3 x 3 x 3 ⇒ # weights = 27

For RGB image the dimension is width x height x depth(channel)

## # of kernels

- A conv layer has multiple kernels
- Each kernel outputs a single 2D array
- Output from a layer has as many 2D arrays as # of kernels

## Pooling

- Downsample the image (Shrink image)
- Overlaying grid on image
- Max/average pooling
- No parameters

### Pooling settings

- Grid size
- Stride
- Type (e.g., max, average)

### Pooling example

- Max pooling (2x2, stride 2)

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2017.png)

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2018.png)

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2019.png)

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2020.png)

## CNN Architecture

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2021.png)

## Applying convolution/pooling to audio

- Spectrogram/MFCC = image
- Time, frequency = x, y
- Amplitude = pixel value

## Preparing MFCCs for a CNN

- 13 MFCCs
- Hop length = 512 samples
- # samples in audio file = 51200

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2022.png)

⇒ Datashape = 100 = (samples / hop length) x 13 = MFCCs x 1 = (depth)

## Recurrent Neural Networks

### RNNs

- Order is important
    - Maintain the understanding on the data's order
- Variable length
- Used for sequential data
- Each item is processed in context
- Ideal for audio/music
    - Audio can be understand as a time interval ⇒ univariate time series ⇒ 1 value, 1 measure take part in each time interval

### Univariate time series

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2023.png)

Dimension [22050x9, 1]

⇒ Each interval is sampled 22050 times

### Multivariate time series

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2024.png)

MFCCs can represent the multivariate time series

For each interval, we have 13 values ⇒ each value is a MFCC-coefficients

[sr/hop_length x 9, #MFCCs] = [387, 13]

### Intuition

- Input data points one at a time
- Predict next step
- Prediction depends on previous data points

### RNN Architecture

![untitled.png](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/untitled.png)

**Recurrent layer**

shape of X = [batch size, # steps, # dimensions]

![RNN_recurrent_layer.png](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/RNN_recurrent_layer.png)

Cell is the one processing the information (sequential data)

Input data represents in Xt

Output is Ht ⇒ Hidden state vector (represent the memory of the network of the certain point of time)

Output is Yt ⇒ The actual output

The Ht is reused later for the next step

⇒ Ht giving the information about the context for later time step

**Unrolling a recurrent layer**

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2025.png)

1 cell used recursively on the time intervals

### Data shape

[batch size, # steps, # dimensions] = [2, 9, 1]

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2026.png)

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2027.png)

⇒ Output shape = [2, 9 ,3] = [batch size, # steps, # units]

In the simple RNN ⇒ Ht shape = Yt shape

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2028.png)

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2029.png)

### Sequence to vector RNN

Only wait for the output on the last time step, dropping all the output before

E.g., Generating melody, only wait for the last note, not necessary care about the previous notes

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2030.png)

### Sequence to sequence RNN

Feed the input batch of sequences

⇒ Fetching the output batch of sequences

Simple RNN is usually sequence to vector rather than sequence to sequence 

### Memory cell for simple RNN

- Dense layer
- Input = state vector + input data
- Activation function = tanh (hyperbolic tangent)

**Why using *tanh* as activation function**

- Training RNNs is difficult
- Vanishing gradients + exploding gradients
    - Vanishing: Gradients tend to disappear
    - Exploding: gradients tend to go bigger and bigger
- RELU can explode (RELU aren't bounded)
- *tanh* maintains values in [-1, 1]

### Backpropagation through time (BPTT)

- Error is back propagated through time
    - Each time step can be seen as a layer in a feed forward network
- Very deep network! ⇒ Vanishing gradient
- RNN is unrolled and treated as a feedforward network

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2031.png)

Output prediction at each time step (comparing with the target)

Calculate the error at each time step and back propagate the error 

⇒ Stabilizing the training process

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2032.png)

At the end after training we can just drop all the previous layers and only used the last layer

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2033.png)

### The math behind

![RNN_bptt.png](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/RNN_bptt.png)

$h_t = f(Ux_t+Wh_{t-1})$

$x_t$: current input

$h_{t-1}$: state vector of previous step

$y_t = softmax(Vh_t)$

In the RNN, we are learning the $U$, $W$ and $V$

### Issues with the simple RNNs

- No long-term memory
    - They can't be used to see much into the past
- Network can't use info from the distant past
- Can't learn patterns with long dependencies

⇒ Long short term memory (LSTM) used to solve these issues

## Long short term memory (LSTM)

- Special type of RNN
- Can learn long-term patterns
- Detects patterns up to 100 steps
- Struggles with 100s/1000s of steps

### RNN vs LSTM

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2034.png)

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2035.png)

### LSTM cell

- Contains a simple RNN cell (the tanh)
- Second state vector = cell state = long-term memory
- Forget gate
- Input gate
- Output gate
    - All three types of gates are connected to a dense sigmoid layer
- Gates works as filters

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2036.png)

$h_t$ the output

$h_t$ hidden state = output

$c_t$ cell state

$x_t$ input

**simple RNN cell** are the one with the tanh (a dense layer with tanh as activation function) 

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2037.png)

**Short-term memory / hidden state**

Keeps information happening in the current state and store that kind of info

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2038.png)

**The cell state** are responsible for long-term memory

- Cell state are updated twice (the multiple and the addition)
- Few computations ⇒ stabilize the gradients ⇒ avoiding vanishing gradients

The multiplication one decide what to forget

The addition one decide what new info to remember

Keep track of the most important info, dropping the less important info

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2039.png)

**Forget segment**

$f_t = \sigma(W_f[h_{t-1}, x_t]+ b_f)$

$f_t$ the forget matrix for time t ⇒ the result of the forget gate ($\sigma$ dense layer)

$b_f$ bias term

- Closer to 0 ⇒ Closer to forgetting
- 1 is to remember

$C^f_t = C_{t-1} * f_t$ ⇒ Cell state of previous time step that we forgetting in this time step

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2040.png)

**Input segment**

Includes the simple RNN cell and input gate (sigmoid dense layer)

$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$ ⇒ Act as a filter on the simple RNN cell

$C'_t = tanh(W_c[h_{t-1}, x_t] + b_C)$

⇒ $C_t^i = C'_t * i_t$ → The cell state at time t for the input

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2041.png)

**Cell state at current time step**

$C_t = C^f_t + C^i_t$

red = purple + blue

$C^f_t$ What to forget and add the what to add as new information $C^i_t$

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2042.png)

**Output segment**

$o_t = \sigma(W_O[h_{t-1}, x_t] + b_O)$

$h_t = o_t * tanh(C_t)$

$tanh$ in this case is just a function to apply to the $C_t$

$h_t$ in this case will be used as hidden state layer for the next time step and for the output for the prediction

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2043.png)

### LSTM variants

- Gated recurrent Unit - GRU

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2044.png)

## Implementation and result on 3 implementations on the same problem (Music genre classification)

All of the three model are running and compiling on the same section, meaning using the `train_test_split` test set and validation set are the same for 3 implementations

Test set's size is 0.25 of the training set's size and validation set's size is 0.2 of the remaining training test size (0.75 of the original size).

All of the models having the same output shape, same number of epochs (30) and batch size (32)

Optimizer used is Adam with learning rate 0.0001 for all 3 models

Original dataset can be found here:  [marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html)

- With 10 genres, each genre consists of 100 30s-length songs

The extracted MFCC dataset file can be found here: [https://1drv.ms/u/s!AmPIeVWwqevugsMlqtz2JmCONdo15w?e=JhY0U0](https://1drv.ms/u/s!AmPIeVWwqevugsMlqtz2JmCONdo15w?e=JhY0U0)

- Each of the sample of the dataset is 3 seconds long, MFCCs extracted with hop_length of 512, n_mfcc=13 and n_fft=2048, sample rate used 22050 thus the shape of each MFCCs extracted from each sample should be [130, 13] ⇒ 130 = 22050*3/hop_length and 13 = n_mfcc

### Feed forward neural network (with overfitting solving)

**Model implementation:**

```python
def build_nn_model(input_shape):
    """Generate NN model

        :param: input_shape (tuple): shape of the input
        :return model: NN model
    """    
    model = keras.Sequential([
        # input layer
        # multi demensional array and flatten it out
        keras.layers.Flatten(input_shape=input_shape),
        # 1st hidden layer
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # output layer
        # softmax: the sum of the result of all the labels = 1
        # predicting: pick the neuron hav highest value
        keras.layers.Dense(10, activation="softmax")
    ])

    return model
```

**History of the train-test accuracy and error through epochs**

![nn_history.png](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/nn_history.png)

### CNN

**Model implementation**

```python
def build_cnn_model(input_shape):
    """Generate CNN model

        :param: input_shape (tuple): shape of the input
        :return model: CNN model
    """    
    # create model
    model = keras.Sequential()

    # 1st convolution layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same')) # padding='same' is zero-padding
    model.add(keras.layers.BatchNormalization()) # standadized the activation layer -> speed up training process

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
```

**History of the train-test accuracy and error through epochs**

![cnn_history.png](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/cnn_history.png)

### RNN and LSTM neural network

**Model implementation:**

```python
def build_lstm_model(input_shape):
    """Generate RNN-LSTM model

        :param: input_shape (tuple): shape of the input
        :return model: RNN-LSTM model
    """
    
    # create model
    model = keras.Sequential()

    # 2 LSTM layers
    # return_sequences = TRUE => sequence to sequence RNN layer
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True)) 
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
```

**History of the train-test accuracy and error through epochs**

![lstm_history.png](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/lstm_history.png)

Note: The test error and test accuracy in all three models indicate the error and accuracy on the validation using the validation set (25% of the original training set)

**Test accuracy result:**

![test_result.png](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/test_result.png)

As shown on all the history and the test accuracy results, the CNN yielded the most accuracy (~0.73). Thus concludes the effectiveness of using CNN on MFCCs features. 

### Extra:

The log on the train and test process

```python
Using TensorFlow backend.
2021-11-05 07:46:34.897940: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
2021-11-05 07:46:53.040240: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2021-11-05 07:46:53.394572: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce 940MX computeCapability: 5.0
coreClock: 0.8605GHz coreCount: 4 deviceMemorySize: 2.00GiB deviceMemoryBandwidth: 37.33GiB/s
2021-11-05 07:46:53.395233: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
2021-11-05 07:46:53.400293: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_10.dll       
2021-11-05 07:46:53.405717: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cufft64_10.dll
2021-11-05 07:46:53.408886: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library curand64_10.dll       
2021-11-05 07:46:53.414959: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusolver64_10.dll     
2021-11-05 07:46:53.418164: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusparse64_10.dll     
2021-11-05 07:46:53.430158: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2021-11-05 07:46:53.430649: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2021-11-05 07:46:53.431606: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2021-11-05 07:46:53.433483: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: NVIDIA GeForce 940MX computeCapability: 5.0
coreClock: 0.8605GHz coreCount: 4 deviceMemorySize: 2.00GiB deviceMemoryBandwidth: 37.33GiB/s
2021-11-05 07:46:53.433990: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll      
2021-11-05 07:46:53.434210: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_10.dll       
2021-11-05 07:46:53.434420: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cufft64_10.dll        
2021-11-05 07:46:53.434648: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library curand64_10.dll       
2021-11-05 07:46:53.434873: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusolver64_10.dll     
2021-11-05 07:46:53.435081: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusparse64_10.dll     
2021-11-05 07:46:53.435300: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2021-11-05 07:46:53.435658: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2021-11-05 07:46:54.530631: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-05 07:46:54.530921: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
2021-11-05 07:46:54.531123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N
2021-11-05 07:46:54.531547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce 940MX, pci bus id: 0000:01:00.0, compute capability: 5.0)
[NN] Model compiling: 
Train on 5991 samples, validate on 1498 samples
Epoch 1/30
2021-11-05 07:46:55.774747: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_10.dll
5991/5991 [==============================] - 3s 443us/step - loss: 25.0164 - accuracy: 0.1521 - val_loss: 4.1317 - val_accuracy: 0.1876
Epoch 2/30
5991/5991 [==============================] - 2s 345us/step - loss: 7.7533 - accuracy: 0.1698 - val_loss: 3.4575 - val_accuracy: 0.1862
Epoch 3/30
5991/5991 [==============================] - 2s 339us/step - loss: 5.0756 - accuracy: 0.1469 - val_loss: 3.4758 - val_accuracy: 0.1435
Epoch 4/30
5991/5991 [==============================] - 2s 339us/step - loss: 4.1860 - accuracy: 0.1522 - val_loss: 3.4871 - val_accuracy: 0.1362
Epoch 5/30
5991/5991 [==============================] - 2s 344us/step - loss: 3.8263 - accuracy: 0.1627 - val_loss: 3.4631 - val_accuracy: 0.1502
Epoch 6/30
5991/5991 [==============================] - 2s 348us/step - loss: 3.7245 - accuracy: 0.1532 - val_loss: 3.4604 - val_accuracy: 0.1509
Epoch 7/30
5991/5991 [==============================] - 2s 341us/step - loss: 3.6303 - accuracy: 0.1656 - val_loss: 3.4271 - val_accuracy: 0.1682
Epoch 8/30
5991/5991 [==============================] - 2s 339us/step - loss: 3.5635 - accuracy: 0.1651 - val_loss: 3.4217 - val_accuracy: 0.1649
Epoch 9/30
5991/5991 [==============================] - 2s 340us/step - loss: 3.4959 - accuracy: 0.1846 - val_loss: 3.3698 - val_accuracy: 0.2150
Epoch 10/30
5991/5991 [==============================] - 2s 338us/step - loss: 3.4699 - accuracy: 0.1854 - val_loss: 3.2849 - val_accuracy: 0.2664
Epoch 11/30
5991/5991 [==============================] - 2s 347us/step - loss: 3.4073 - accuracy: 0.2170 - val_loss: 3.3021 - val_accuracy: 0.2276
Epoch 12/30
5991/5991 [==============================] - 2s 337us/step - loss: 3.3704 - accuracy: 0.2283 - val_loss: 3.2404 - val_accuracy: 0.2623
Epoch 13/30
5991/5991 [==============================] - 2s 341us/step - loss: 3.3332 - accuracy: 0.2382 - val_loss: 3.1621 - val_accuracy: 0.2944
Epoch 14/30
5991/5991 [==============================] - 2s 340us/step - loss: 3.3104 - accuracy: 0.2504 - val_loss: 3.1673 - val_accuracy: 0.2904
Epoch 15/30
5991/5991 [==============================] - 2s 341us/step - loss: 3.2783 - accuracy: 0.2597 - val_loss: 3.1388 - val_accuracy: 0.2997
Epoch 16/30
5991/5991 [==============================] - 2s 340us/step - loss: 3.2425 - accuracy: 0.2729 - val_loss: 3.0685 - val_accuracy: 0.3244
Epoch 17/30
5991/5991 [==============================] - 2s 354us/step - loss: 3.2261 - accuracy: 0.2654 - val_loss: 3.0620 - val_accuracy: 0.3097
Epoch 18/30
5991/5991 [==============================] - 2s 350us/step - loss: 3.1965 - accuracy: 0.2719 - val_loss: 3.1032 - val_accuracy: 0.2844
Epoch 19/30
5991/5991 [==============================] - 2s 346us/step - loss: 3.1364 - accuracy: 0.2913 - val_loss: 3.0472 - val_accuracy: 0.3064
Epoch 20/30
5991/5991 [==============================] - 2s 340us/step - loss: 3.1227 - accuracy: 0.2859 - val_loss: 3.0135 - val_accuracy: 0.3238
Epoch 21/30
5991/5991 [==============================] - 2s 337us/step - loss: 3.0938 - accuracy: 0.3065 - val_loss: 3.0068 - val_accuracy: 0.3204
Epoch 22/30
5991/5991 [==============================] - 2s 341us/step - loss: 3.0685 - accuracy: 0.3055 - val_loss: 2.9713 - val_accuracy: 0.3385
Epoch 23/30
5991/5991 [==============================] - 2s 341us/step - loss: 3.0370 - accuracy: 0.3123 - val_loss: 2.9389 - val_accuracy: 0.3431
Epoch 24/30
5991/5991 [==============================] - 2s 347us/step - loss: 2.9946 - accuracy: 0.3223 - val_loss: 2.9232 - val_accuracy: 0.3391
Epoch 25/30
5991/5991 [==============================] - 2s 340us/step - loss: 2.9819 - accuracy: 0.3145 - val_loss: 2.9082 - val_accuracy: 0.3478
Epoch 26/30
5991/5991 [==============================] - 2s 338us/step - loss: 2.9741 - accuracy: 0.3238 - val_loss: 2.8907 - val_accuracy: 0.3531
Epoch 27/30
5991/5991 [==============================] - 2s 337us/step - loss: 2.9452 - accuracy: 0.3252 - val_loss: 2.9031 - val_accuracy: 0.3445
Epoch 28/30
5991/5991 [==============================] - 2s 338us/step - loss: 2.9034 - accuracy: 0.3268 - val_loss: 2.8288 - val_accuracy: 0.3625
Epoch 29/30
5991/5991 [==============================] - 2s 342us/step - loss: 2.8895 - accuracy: 0.3352 - val_loss: 2.7727 - val_accuracy: 0.3618
Epoch 30/30
5991/5991 [==============================] - 2s 342us/step - loss: 2.8338 - accuracy: 0.3332 - val_loss: 2.7797 - val_accuracy: 0.3732
[CNN] Model compiling: 
Train on 5991 samples, validate on 1498 samples
Epoch 1/30
2021-11-05 07:48:59.933304: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2021-11-05 07:49:07.239692: W tensorflow/stream_executor/gpu/redzone_allocator.cc:312] Internal: Invoking GPU asm compilation is supported on Cuda non-Windows platforms only
Relying on driver to perform ptx compilation. This message will be only logged once.
5991/5991 [==============================] - 17s 3ms/step - loss: 1.8575 - accuracy: 0.3679 - val_loss: 1.5314 - val_accuracy: 0.4419
Epoch 2/30
5991/5991 [==============================] - 5s 778us/step - loss: 1.4762 - accuracy: 0.4799 - val_loss: 1.2921 - val_accuracy: 0.5340
Epoch 3/30
5991/5991 [==============================] - 4s 726us/step - loss: 1.3757 - accuracy: 0.5098 - val_loss: 1.2104 - val_accuracy: 0.5701
Epoch 4/30
5991/5991 [==============================] - 4s 644us/step - loss: 1.2805 - accuracy: 0.5447 - val_loss: 1.1492 - val_accuracy: 0.5834
Epoch 5/30
5991/5991 [==============================] - 4s 633us/step - loss: 1.2108 - accuracy: 0.5625 - val_loss: 1.1195 - val_accuracy: 0.5941
Epoch 6/30
5991/5991 [==============================] - 4s 655us/step - loss: 1.1548 - accuracy: 0.5924 - val_loss: 1.1067 - val_accuracy: 0.6035
Epoch 7/30
5991/5991 [==============================] - 4s 664us/step - loss: 1.1268 - accuracy: 0.5986 - val_loss: 1.0592 - val_accuracy: 0.6275
Epoch 8/30
5991/5991 [==============================] - 4s 640us/step - loss: 1.0935 - accuracy: 0.6206 - val_loss: 1.0302 - val_accuracy: 0.6342
Epoch 9/30
5991/5991 [==============================] - 4s 644us/step - loss: 1.0444 - accuracy: 0.6308 - val_loss: 0.9997 - val_accuracy: 0.6495
Epoch 10/30
5991/5991 [==============================] - 4s 643us/step - loss: 1.0124 - accuracy: 0.6450 - val_loss: 0.9855 - val_accuracy: 0.6636
Epoch 11/30
5991/5991 [==============================] - 4s 640us/step - loss: 0.9895 - accuracy: 0.6526 - val_loss: 0.9707 - val_accuracy: 0.6542
Epoch 12/30
5991/5991 [==============================] - 4s 631us/step - loss: 0.9690 - accuracy: 0.6513 - val_loss: 0.9541 - val_accuracy: 0.6722
Epoch 13/30
5991/5991 [==============================] - 4s 633us/step - loss: 0.9465 - accuracy: 0.6662 - val_loss: 0.9344 - val_accuracy: 0.6749
Epoch 14/30
5991/5991 [==============================] - 4s 630us/step - loss: 0.9084 - accuracy: 0.6795 - val_loss: 0.9339 - val_accuracy: 0.6736
Epoch 15/30
5991/5991 [==============================] - 4s 632us/step - loss: 0.8973 - accuracy: 0.6885 - val_loss: 0.9022 - val_accuracy: 0.6776
Epoch 16/30
5991/5991 [==============================] - 4s 632us/step - loss: 0.8646 - accuracy: 0.6914 - val_loss: 0.8980 - val_accuracy: 0.6849
Epoch 17/30
5991/5991 [==============================] - 4s 631us/step - loss: 0.8388 - accuracy: 0.7079 - val_loss: 0.8922 - val_accuracy: 0.6916
Epoch 18/30
5991/5991 [==============================] - 4s 630us/step - loss: 0.8279 - accuracy: 0.7081 - val_loss: 0.8893 - val_accuracy: 0.6862
Epoch 19/30
5991/5991 [==============================] - 4s 637us/step - loss: 0.8219 - accuracy: 0.7131 - val_loss: 0.8683 - val_accuracy: 0.6969
Epoch 20/30
5991/5991 [==============================] - 4s 630us/step - loss: 0.7916 - accuracy: 0.7166 - val_loss: 0.8718 - val_accuracy: 0.6849
Epoch 21/30
5991/5991 [==============================] - 4s 640us/step - loss: 0.7894 - accuracy: 0.7248 - val_loss: 0.8758 - val_accuracy: 0.6896
Epoch 22/30
5991/5991 [==============================] - 4s 632us/step - loss: 0.7536 - accuracy: 0.7359 - val_loss: 0.8458 - val_accuracy: 0.6989
Epoch 23/30
5991/5991 [==============================] - 4s 642us/step - loss: 0.7402 - accuracy: 0.7394 - val_loss: 0.8145 - val_accuracy: 0.7236
Epoch 24/30
5991/5991 [==============================] - 4s 681us/step - loss: 0.7368 - accuracy: 0.7466 - val_loss: 0.8192 - val_accuracy: 0.7029
Epoch 25/30
5991/5991 [==============================] - 4s 737us/step - loss: 0.7136 - accuracy: 0.7485 - val_loss: 0.8283 - val_accuracy: 0.7076
Epoch 26/30
5991/5991 [==============================] - 5s 912us/step - loss: 0.7014 - accuracy: 0.7521 - val_loss: 0.8461 - val_accuracy: 0.7009
Epoch 27/30
5991/5991 [==============================] - 5s 797us/step - loss: 0.6759 - accuracy: 0.7675 - val_loss: 0.8309 - val_accuracy: 0.7103
Epoch 28/30
5991/5991 [==============================] - 5s 847us/step - loss: 0.6785 - accuracy: 0.7626 - val_loss: 0.8669 - val_accuracy: 0.6943
Epoch 29/30
5991/5991 [==============================] - 4s 639us/step - loss: 0.6564 - accuracy: 0.7690 - val_loss: 0.8016 - val_accuracy: 0.7156
Epoch 30/30
5991/5991 [==============================] - 4s 642us/step - loss: 0.6394 - accuracy: 0.7707 - val_loss: 0.8142 - val_accuracy: 0.7156
[LSTM] Model compiling: 
Train on 5991 samples, validate on 1498 samples
Epoch 1/30
5991/5991 [==============================] - 46s 8ms/step - loss: 1.9142 - accuracy: 0.3171 - val_loss: 1.6141 - val_accuracy: 0.4306
Epoch 2/30
5991/5991 [==============================] - 45s 8ms/step - loss: 1.5792 - accuracy: 0.4320 - val_loss: 1.4644 - val_accuracy: 0.4726
Epoch 3/30
5991/5991 [==============================] - 46s 8ms/step - loss: 1.4646 - accuracy: 0.4670 - val_loss: 1.3990 - val_accuracy: 0.4773
Epoch 4/30
5991/5991 [==============================] - 45s 8ms/step - loss: 1.3862 - accuracy: 0.5056 - val_loss: 1.3495 - val_accuracy: 0.5033
Epoch 5/30
5991/5991 [==============================] - 45s 7ms/step - loss: 1.3444 - accuracy: 0.5253 - val_loss: 1.3408 - val_accuracy: 0.5200
Epoch 6/30
5991/5991 [==============================] - 46s 8ms/step - loss: 1.2946 - accuracy: 0.5421 - val_loss: 1.3105 - val_accuracy: 0.5327
Epoch 7/30
5991/5991 [==============================] - 45s 8ms/step - loss: 1.2604 - accuracy: 0.5557 - val_loss: 1.2760 - val_accuracy: 0.5367
Epoch 8/30
5991/5991 [==============================] - 45s 7ms/step - loss: 1.2270 - accuracy: 0.5653 - val_loss: 1.2740 - val_accuracy: 0.5427
Epoch 9/30
5991/5991 [==============================] - 45s 8ms/step - loss: 1.1922 - accuracy: 0.5805 - val_loss: 1.2643 - val_accuracy: 0.5514
Epoch 10/30
5991/5991 [==============================] - 50s 8ms/step - loss: 1.1539 - accuracy: 0.5922 - val_loss: 1.2539 - val_accuracy: 0.5507
Epoch 11/30
5991/5991 [==============================] - 44s 7ms/step - loss: 1.1501 - accuracy: 0.5997 - val_loss: 1.2385 - val_accuracy: 0.5474
Epoch 12/30
5991/5991 [==============================] - 51s 8ms/step - loss: 1.1099 - accuracy: 0.6154 - val_loss: 1.2271 - val_accuracy: 0.5648
Epoch 13/30
5991/5991 [==============================] - 45s 8ms/step - loss: 1.0873 - accuracy: 0.6249 - val_loss: 1.2230 - val_accuracy: 0.5694
Epoch 14/30
5991/5991 [==============================] - 45s 8ms/step - loss: 1.0705 - accuracy: 0.6353 - val_loss: 1.2071 - val_accuracy: 0.5748
Epoch 15/30
5991/5991 [==============================] - 45s 7ms/step - loss: 1.0556 - accuracy: 0.6431 - val_loss: 1.2222 - val_accuracy: 0.5581
Epoch 16/30
5991/5991 [==============================] - 45s 8ms/step - loss: 1.0342 - accuracy: 0.6428 - val_loss: 1.2190 - val_accuracy: 0.5741
Epoch 17/30
5991/5991 [==============================] - 45s 8ms/step - loss: 1.0234 - accuracy: 0.6481 - val_loss: 1.1915 - val_accuracy: 0.5721
Epoch 18/30
5991/5991 [==============================] - 44s 7ms/step - loss: 1.0026 - accuracy: 0.6593 - val_loss: 1.2249 - val_accuracy: 0.5821
Epoch 19/30
5991/5991 [==============================] - 50s 8ms/step - loss: 0.9914 - accuracy: 0.6560 - val_loss: 1.2169 - val_accuracy: 0.5688
Epoch 20/30
5991/5991 [==============================] - 45s 8ms/step - loss: 0.9676 - accuracy: 0.6762 - val_loss: 1.1771 - val_accuracy: 0.5915
Epoch 21/30
5991/5991 [==============================] - 44s 7ms/step - loss: 0.9502 - accuracy: 0.6797 - val_loss: 1.1975 - val_accuracy: 0.5821
Epoch 22/30
5991/5991 [==============================] - 45s 8ms/step - loss: 0.9399 - accuracy: 0.6752 - val_loss: 1.1826 - val_accuracy: 0.5928
Epoch 23/30
5991/5991 [==============================] - 45s 8ms/step - loss: 0.9237 - accuracy: 0.6940 - val_loss: 1.1812 - val_accuracy: 0.5981
Epoch 24/30
5991/5991 [==============================] - 45s 7ms/step - loss: 0.9093 - accuracy: 0.6940 - val_loss: 1.1900 - val_accuracy: 0.6008
Epoch 25/30
5991/5991 [==============================] - 50s 8ms/step - loss: 0.9013 - accuracy: 0.6965 - val_loss: 1.1930 - val_accuracy: 0.5908
Epoch 26/30
5991/5991 [==============================] - 45s 8ms/step - loss: 0.8859 - accuracy: 0.7007 - val_loss: 1.1838 - val_accuracy: 0.6142
Epoch 27/30
5991/5991 [==============================] - 51s 8ms/step - loss: 0.8761 - accuracy: 0.7062 - val_loss: 1.1564 - val_accuracy: 0.6021
Epoch 28/30
5991/5991 [==============================] - 44s 7ms/step - loss: 0.8639 - accuracy: 0.7102 - val_loss: 1.1870 - val_accuracy: 0.5975
Epoch 29/30
5991/5991 [==============================] - 45s 8ms/step - loss: 0.8721 - accuracy: 0.7091 - val_loss: 1.1957 - val_accuracy: 0.5935
Epoch 30/30
5991/5991 [==============================] - 45s 8ms/step - loss: 0.8526 - accuracy: 0.7156 - val_loss: 1.1368 - val_accuracy: 0.6055
[NN] Test accuracy: 0.39487385749816895
[CNN] Test accuracy: 0.7324789762496948
[LSTM] Test accuracy: 0.6487785577774048
```

## Note taken:

- Find a good dataset and validate the model on those
- Then statistically validate it

![Untitled](Deep%20learning%20(for%20audio)%20with%20Python%20afd6fc842d8d4c86a3aa6499113d8526/Untitled%2045.png)

→ The valence arousal calculated was also based on the annotated valence and arousal

→ Not using the dataset anymore, but the methodology still true so I kept it there.

## Works to do later:

- Find a dataset fit for the usage of the music emotion recognition (Currently intent to use the [Million Song Dataset](http://millionsongdataset.com/))
- Fetch the raw audio signal and extract the needed MFCCs for the model
- Build a music emotion recognition model based on the input generated
- Tweaking the performance
- Write some reports on it
- Convert this report to pdf format also

## References

[Understanding LSTM Networks -- colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf](https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf)

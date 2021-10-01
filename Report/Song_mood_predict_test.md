# Predicting the Music Mood of a Song with Deep Learning.

The steps and models tutorial was taken from the article: 

[Predicting the Music Mood of a Song with Deep Learning.](https://towardsdatascience.com/predicting-the-music-mood-of-a-song-with-deep-learning-c3ac2b45229e)

This report includes the parts that needed and my take on understanding the implementation

## Features to use:

Extracted from Spotify API

- **Acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- **Danceability**: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- **Energy**: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
- **Instrumentalness**: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
- **Liveness**: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides a strong likelihood that the track is live.
- **Loudness**: the overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing the relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
- **Speechiness**: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audiobook, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- **Valence**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- **Tempo**: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, the tempo is the speed or pace of a given piece and derives directly from the average beat duration.

## Building the model

### Preprocessing data

`MinMaxScaler` is used to scale the features' value into the value between [0,1] and preserving the shape of the original distribution. The label also encoded. Finally, the data set is splited 80% for training and 20% for testing.

```python
col_features = df.columns[6:-2]
print((col_features))
# Scale the value
X= MinMaxScaler().fit_transform(df[col_features])
X2 = np.array(df[col_features])
Y = df['mood']

# Encode the label
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)

#Split train and test data with a test size of 20%
X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)
```

### Creating the model

The model consists of a Multi-Class Neural Network with an input of 13 Features in my case, 1 Layer with 8 nodes (will try to change this and see the effect and have details report later), and 4 outputs on the output Layer.

As stated in the article, KerasClassifier is used as the estimator. The activation Function was Rectified Linear Unit (Relu), the Loss function is a Logistic Function and Adam Gradient Descent Algorithm is the optimizer.

I tried to use Stochastic Gradient Descent (sgd) as recommended (walkthroughed some stackoverflow's post) and it yielded better result for the author's dataset.

The declaration of the model:

```python
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)

def base_model():
    #Create the model
    model = Sequential()
    #Add 1 layer with 8 nodes,input of 4 dim with relu function
    model.add(Dense(8,input_dim=11,activation='relu'))
    #Add 1 layer with output of 4 and softmax activation function
    model.add(Dense(4,activation='softmax'))
    #Compile the model using sigmoid loss function and adam optim
    # sgd optimizer improve the accuracy on the author's dataset
    model.compile(loss='categorical_crossentropy',optimizer=sgd,
                 metrics=['accuracy'])
    return model
```

The estimator (currently trying to opting the epochs and batch_size for better result, still need to understand the model more thoroughly though)

```python
estimator = KerasClassifier(build_fn=base_model,epochs=30,batch_size=64,verbose=1)
```

### Model evaluation

Using **K-Fold Cross Validation** to evalute the estimator using the train data. The number of splits is K=**10** shuffling all the values.

```python
kfold = KFold(n_splits=10,shuffle=True)
results = cross_val_score(estimator,X,encoded_y,cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))
```

### Training the model

```python
estimator.fit(X_train,Y_train)
y_preds = estimator.predict(X_test)
```

That's it for today. The report will be updated soon with more info (hopefully) and some conclusion on what have been done!

Note:

- Tried to train the dataset Minh gives (without the `time_signature` and `mode` features), the accuracy yields only for 50%, maybe it needs improvement
- Learned about epochs, batch size during the model test
- Will look into it more

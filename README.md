# DecodeEEG

### Dataset used - [A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758#sec0006)

## Step 1 -  Download dataset

- Download the files under folder `Preprocessed EEG data` and `Image set` from the database
- Unzip and store them in the following structure

```
  project/
├── data/
│   ├── PreprocessedEEG/           % Preprocessed EEG data subject wise
|   |   ├──sub-01
|   |   ├──sub-02
|   |    ...
│   ├── images/                    % images
|   |   ├──training_images
|   |   ├──test_images

```

## Step 2 - Preprocessing the images to feature vectors

Run the below scripts to extract feature vectors from training and test images

Model used - ResNet18

```
cd src/img_preprocessing
python3 extract_features.py --train true --test true
```

Extracted vectors are stored in folder `data/feature_vectors/training` and `data/feature_vectors/test`


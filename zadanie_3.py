import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mahotas.features.texture as mhtex
from sklearn.preprocessing import LabelEncoder


def extract_texture_samples(input_dir, output_dir, sample_size):
    for root, dirs, files in os.walk(input_dir):
        print("Przetwarzany katalog:", root)
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                print("Przetwarzany obraz:", image_path)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                height, width = image.shape

                category = os.path.basename(root)

                output_category_dir = os.path.join(output_dir, category)
                os.makedirs(output_category_dir, exist_ok=True)

                for i in range(0, height - sample_size[0], sample_size[0]):
                    for j in range(0, width - sample_size[1], sample_size[1]):
                        sample = image[i:i+sample_size[0], j:j+sample_size[1]]
                        cv2.imwrite(os.path.join(output_category_dir, f"{file}_{i}_{j}.jpg"), sample)

def calculate_texture_features(input_dir, distances=[1, 3, 5], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    features_list = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                glcm = mhtex.haralick(image.astype(np.uint8))

                features = {'file': file, 'category': os.path.basename(root).split('_')[0]}  # Użyj części nazwy katalogu jako kategorii

                for prop in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']:
                    for dist in distances:
                        for ang in angles:
                            feature_key = f'{prop}_d{dist}_a{int(np.degrees(ang))}'
                            feature_value = mhtex.haralick(image.astype(np.uint8))[distances.index(dist), angles.index(ang)]
                            features[feature_key] = feature_value

                features_list.append(features)

    features_df = pd.DataFrame(features_list)
    print(f"Przetworzono {len(features_df)} próbek.")
    return features_df


def save_features_to_csv(features, output_file):
    df = pd.DataFrame(features)
    df.to_csv(output_file, index=False)


def classify_features(features_file, test_size=0.2):
    df = pd.read_csv(features_file)

    df = df.drop('file', axis=1)

    X = df.drop('category', axis=1)
    y = df['category']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_test = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print("Dokładność klasyfikatora:", accuracy)

input_dir = r'C:\Users\Dell T3500\Desktop\struktury'  # Katalog z oryginalnymi obrazami
output_dir = r'C:\Users\Dell T3500\Desktop\wycinki'  # Katalog, w którym zostaną zapisane wycinki
features_file = 'texture_features.csv'

sample_size = (128, 128)
extract_texture_samples(input_dir, output_dir, sample_size)

texture_features = calculate_texture_features(output_dir)
save_features_to_csv(texture_features, features_file)

classify_features(features_file)


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\Dell T3500\PycharmProjects\POI1\texture_features.csv", sep=',')

# Usunięcie kolumn 'plik' i 'kategoria', konwersja danych do float
X = df.drop(['file', 'category'], axis=1).astype('float').values

# Etykiety kategorii
y = df['category'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)

# Definicja modelu
model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='sigmoid'))  # Pierwsza warstwa ukryta
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Warstwa wyjściowa

# Kompilacja modelu
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Wyświetlenie informacji o modelu
model.summary()

# Trenowanie modelu
model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)

# Przewidywanie na zbiorze testowym
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Tworzenie macierzy pomyłek
cm = confusion_matrix(y_test, y_pred_classes)
print(cm)

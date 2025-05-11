import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

st.title("🧠 CNN for MNIST Digit Classification")
st.write("Upload a CSV file in MNIST format (like `mnist.csv`) to train and visualize predictions.")

uploaded_file = st.file_uploader("Upload MNIST CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Data Preparation
    X = data.drop('label', axis=1).values / 255.0
    y = data['label'].values
    X = X.reshape(-1, 28, 28, 1)
    y_cat = to_categorical(y, 10)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    # Model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)

    # Train
    st.write("Training model... please wait.")
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1,
                        callbacks=[early_stop, reduce_lr], verbose=0)

    # Evaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"✅ Test Accuracy: {acc:.4f}")

    # Confusion Matrix
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Show 25 predictions
    st.subheader("🔍 Sample Predictions (25 images)")
    fig2, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_true[i]}")
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig2)

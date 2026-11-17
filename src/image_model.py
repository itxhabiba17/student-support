"""Small CNN for 28x28x1 digit images (MNIST subset).
TensorFlow is imported lazily so the other tabs work even without it."""
import numpy as np

from .config import MODELS, SEED

MODEL_PATH = MODELS / "images.keras"
IMG_SHAPE = (28, 28, 1)


def build_model():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=IMG_SHAPE),
        tf.keras.layers.Rescaling(1. / 255),          # arrays are 0-255; do NOT divide again
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def train(images, epochs=3):
    import tensorflow as tf
    model = build_model()
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1,
                                            restore_best_weights=True)
    history = model.fit(images["x_train"], images["y_train"],
                        validation_data=(images["x_validation"], images["y_validation"]),
                        epochs=epochs, batch_size=32, callbacks=[stop], verbose=2)
    model.save(MODEL_PATH)
    return model, history


def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(pil_img, invert="auto") -> np.ndarray:
    """Any PNG/JPG -> (1, 28, 28, 1) float32 in 0-255 (saved colour mode, size, batch dim).
    MNIST digits are white on black; dark-on-white photos are inverted."""
    from PIL import ImageOps
    g = pil_img.convert("L").resize(IMG_SHAPE[:2])
    arr = np.asarray(g, dtype="float32")
    if invert == "auto":
        invert = arr.mean() > 127                   # mostly white background -> invert
    if invert:
        arr = np.asarray(ImageOps.invert(g), dtype="float32")
    return arr.reshape(1, *IMG_SHAPE)


def predict(model, arr):
    scores = model.predict(arr, verbose=0)[0]
    return int(scores.argmax()), scores

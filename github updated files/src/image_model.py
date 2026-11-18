"""CNN for 28x28x1 digit images (MNIST subset).
TensorFlow is imported lazily so the other tabs work even without it."""
import numpy as np

from .config import MODELS, SEED

MODEL_PATH = MODELS / "images.keras"
IMG_SHAPE = (28, 28, 1)


def build_model():
    """Improved CNN used by the notebook (see README: Deviation from the proposal)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=IMG_SHAPE),
        tf.keras.layers.RandomRotation(0.04, fill_mode="constant"),          # training only
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant"), # training only
        tf.keras.layers.Rescaling(1. / 255),          # arrays are 0-255; do NOT divide again
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def train(images, epochs=15):
    import tensorflow as tf
    model = build_model()
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
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
    """Any PNG/JPG -> (1, 28, 28, 1) float32 in 0-255, prepared the way MNIST digits are:
    grey (saved colour mode), white digit on black, digit cropped and scaled to fit a
    20x20 box, then placed in the 28x28 frame with its centre of mass in the middle.
    Values stay 0-255 because the model rescales once itself."""
    from PIL import Image, ImageOps
    g = pil_img.convert("L")
    if invert == "auto":
        invert = np.asarray(g, dtype="float32").mean() > 127   # light background -> invert
    if invert:
        g = ImageOps.invert(g)
    arr = np.asarray(g, dtype="float32")
    if arr.max() <= 0:                                          # blank image
        return np.zeros((1, *IMG_SHAPE), dtype="float32")

    ys, xs = np.where(arr > 0.2 * arr.max())                    # ignore faint background
    g = g.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    w, h = g.size
    scale = 20 / max(w, h)
    g = g.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("L", IMG_SHAPE[:2], 0)
    canvas.paste(g, ((IMG_SHAPE[1] - g.size[0]) // 2, (IMG_SHAPE[0] - g.size[1]) // 2))

    arr = np.asarray(canvas, dtype="float32")
    yy, xx = np.indices(arr.shape)
    total = arr.sum()
    dy = int(round(IMG_SHAPE[0] / 2 - (yy * arr).sum() / total))
    dx = int(round(IMG_SHAPE[1] / 2 - (xx * arr).sum() / total))
    arr = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
    return arr.reshape(1, *IMG_SHAPE)


def predict(model, arr):
    scores = model.predict(arr, verbose=0)[0]
    return int(scores.argmax()), scores

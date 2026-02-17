import os
import shutil
import tensorflow as tf

MODEL_NAME = "incident_classifier"
VERSION = "1"
INPUT_DIM = 128

def build_model():
    inputs = tf.keras.Input(shape=(INPUT_DIM,), dtype=tf.float32, name="x")
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="probs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def export_model():
    model = build_model()

    # weights/graph build için bir kez çağır
    _ = model(tf.zeros([1, INPUT_DIM], dtype=tf.float32))

    export_path = os.path.join("models", MODEL_NAME, VERSION)

    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    # Keras 3: TF Serving için doğru format -> SavedModel bundle üretir
    # (export_path altında saved_model.pb olmalı)
    model.export(export_path)

    print("✅ Export completed:", export_path)
    print("✅ Expect:", os.path.join(export_path, "saved_model.pb"))

if __name__ == "__main__":
    export_model()

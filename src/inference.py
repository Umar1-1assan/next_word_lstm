import numpy as np
import tensorflow as tf
import pickle


def predict_next(seed_text: str,
                 model_path: str = '../models/best_model.h5',
                 tokenizer_path: str = '../data/processed/tokenizer.pkl',
                 top_k: int = 3) -> list[tuple[str, float]]:
    """
    Given a seed_text, predict top_k next words and their probabilities.
    Returns list of (word, prob).
    """
    # Load tokenizer and model
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(model_path)

    # Prepare sequence
    seq = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    seq = np.array(seq[-model.input_shape[1]:])  # trim/pad to seq_length
    seq = np.expand_dims(seq, 0)

    # Predict
    preds = model.predict(seq, verbose=0)[0]
    idxs = np.argsort(preds)[-top_k:][::-1]
    return [(tokenizer.index_word[i], float(preds[i])) for i in idxs]
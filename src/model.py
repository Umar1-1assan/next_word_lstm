import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


def build_model(vocab_size: int,
                seq_length: int,
                embed_dim: int = 100,
                lstm_units: list[int] = [128],
                dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Construct a word-level LSTM model for next-word prediction.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embed_dim,
                        input_length=seq_length,
                        name='embedding'))
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        model.add(LSTM(units, return_sequences=return_seq, name=f'lstm_{i+1}'))
        model.add(Dropout(dropout_rate, name=f'dropout_{i+1}'))
    model.add(Dense(vocab_size, activation='softmax', name='output'))
    return model

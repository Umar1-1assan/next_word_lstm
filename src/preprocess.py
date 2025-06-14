import re
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(txt: str) -> str:
    """
    Clean raw Shakespeare text:
      - Remove stage directions and bracketed text
      - Strip ACT/SCENE headers
      - Keep alphanumeric and basic punctuation
      - Normalize to lowercase and collapse whitespace
    """
    txt = re.sub(r"\[.*?\]", "", txt)
    txt = re.sub(r"\(.*?\)", "", txt)
    txt = re.sub(r"^(ACT|SCENE) [IVXLC]+.*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"[^a-zA-Z0-9\s\.\,\;\'\-]", "", txt)
    txt = txt.lower()
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def load_and_clean(input_path: str) -> str:
    with open(input_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    return clean_text(raw)


def tokenize_and_save(cleaned_text: str,
                      tokenizer_path: str,
                      num_words: int = None) -> Tokenizer:
    tokenizer = Tokenizer(num_words=num_words)
    # we can split into sentences by punctuation
    sentences = cleaned_text.split('.')
    tokenizer.fit_on_texts(sentences)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    return tokenizer


def generate_sequences(tokenizer: Tokenizer,
                       cleaned_text: str,
                       max_len: int = None) -> (np.ndarray, np.ndarray):
    sentences = cleaned_text.split('.')
    sequences = []
    for sent in sentences:
        token_list = tokenizer.texts_to_sequences([sent])[0]
        for i in range(1, len(token_list)):
            seq = token_list[:i+1]
            sequences.append(seq)
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    sequences_padded = pad_sequences(sequences, maxlen=max_len, padding='pre')
    sequences_padded = np.array(sequences_padded)
    X, y = sequences_padded[:, :-1], sequences_padded[:, -1]
    return X, y, max_len


def save_numpy_arrays(X: np.ndarray,
                      y: np.ndarray,
                      x_path: str,
                      y_path: str) -> None:
    np.save(x_path, X)
    np.save(y_path, y)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess Shakespeare text')
    parser.add_argument('--input', default='../data/raw/shakespeare.txt')
    parser.add_argument('--out_dir', default='../data/processed')
    parser.add_argument('--num_words', type=int, default=None)
    args = parser.parse_args()

    clean = load_and_clean(args.input)
    tokenizer = tokenize_and_save(clean,
                                  f"{args.out_dir}/tokenizer.pkl",
                                  num_words=args.num_words)
    X, y, seq_len = generate_sequences(tokenizer, clean)
    save_numpy_arrays(X, y,
                      f"{args.out_dir}/data_X.npy",
                      f"{args.out_dir}/data_y.npy")
    print(f"Saved sequences (num={X.shape[0]}, seq_len={seq_len}) and tokenizer.")
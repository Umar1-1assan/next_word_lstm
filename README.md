**Submitted by:** Muhammad Umar Hassan  
**Program:** B.Sc. Computer Science, FAST NUCES  
**Internship Task:** KDD LAB Gen AI Internship – Next‑Word Prediction Using Word‑Level LSTM on Shakespeare  
**Date:** June 15, 2025  

---

## 1. Introduction

This report documents the design, implementation, and evaluation of a word‑level LSTM model trained on Shakespeare’s plays to perform next‑word prediction. It covers:

- System architecture and data pipeline  
- Qualitative evaluation with example predictions  

GitHub Link:  
[https://github.com/Umar1-1assan/next_word_lstm](https://github.com/Umar1-1assan/next_word_lstm)

---

## 2. System Architecture & Data Pipeline

### 2.1 Overview Diagram

**Preprocessing.ipynb**  
- Filter dialogue lines  
- Clean (remove brackets, normalize, lowercase)  
- Tokenize word‑level  
- Sliding‑window sequence generation  
- **OUTPUT:** `data_X.npy`, `data_y.npy`, `tokenizer.pkl`

**Modeling.ipynb**  
- Build Keras Sequential model: Embedding → LSTM(s) → Dense(softmax)  
- Hyperparameter sweeps  
- Train + checkpoint  
- **OUTPUT:** `best_model.h5`

**Test_user_input.ipynb**  
- Single‑run CLI for next‑word prediction

### 2.2 Data Preprocessing

- **Filtering:** Removed all non‑dialogue rows (ACT/SCENE headings) and any empty lines.  
- **Cleaning:** Stripped out stage directions (`[…]`, `(...)`), non‑alphanumeric characters (except basic punctuation), collapsed whitespace, and converted to lowercase.  
- **Tokenization:** Built a single word‑level `Tokenizer` over the entire cleaned text, yielding a vocabulary of ~`X` words.  
- **Sequence Generation:** Used a sliding window of length *n = 20* to create sequences of 20 input tokens and 1 label.  
- **Outputs:**  
  - `data_X.npy` shape: `(Z, 20)`  
  - `data_y.npy` shape: `(Z,)`  
  - `tokenizer.pkl` (word→index mappings)  

---

## 3. Model Development & Hyperparameter Tuning

### 3.1 Base Architecture

- **Embedding:** 100‑dimensional  
- **LSTM layers:** Two layers of 128 units each  
- **Dropout:** 0.2 after each LSTM  
- **Output:** Dense softmax over the full vocabulary  

```python
    def build_model(vocab_size, seq_length,
                    embed_dim=100,
                    lstm_units=[128, 128],
                    dropout_rate=0.2):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size,
                            output_dim=embed_dim,
                            input_length=seq_length))
        for i, units in enumerate(lstm_units):
            model.add(LSTM(units, return_sequences=(i < len(lstm_units)-1)))
            model.add(Dropout(dropout_rate))
        model.add(Dense(vocab_size, activation='softmax'))
        return model

    model = build_model(vocab_size, seq_length, embed_dim=100, lstm_units=[128,128])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
 ```
 ## 3.2 Training Configuration

- **Epochs:** 10  
- **Batch size:** 128  
- **Validation split:** 10%  
- **Optimizer:** Adam, learning rate = 1e‑3  
- **Callbacks:**  
  - `ModelCheckpoint` (save best model by `val_loss`)  
  - `EarlyStopping` (`patience=5`, `restore_best_weights=True`)  

---

## 4. Evaluation

Run `test_user_input.ipynb` to test any seed phrase:

```text
Enter a seed phrase: to be or not to
Top‑3 predictions:
  be — 3.91%
  that — 1.86%
  the — 1.30%
```

## 5. Conclusions

- **Data Pipeline:** Clean, robust, and reproducible via notebooks + `preprocess.py`.  
- **Model:** `Embedding → 2×LSTM → Dense(softmax)` meets all requirements.  
- **Training:** Steady decrease in loss and increase in accuracy; final top‑1 accuracy ≈ 11.8% on validation.  
- **Hyperparameters:** Best trade‑off found with `seq_len = 20`, `2×128 LSTM`, and `Adam` optimizer at `1e-3`.

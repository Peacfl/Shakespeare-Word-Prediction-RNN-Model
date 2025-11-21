# app.py: Shakespeare Text Generator Streamlit App
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Load exports
@st.cache_resource  # Cache model load for speed
def load_model():
    model = tf.keras.models.load_model('models\word_predictor_shakespeare.keras')
    with open('models\char_to_int.pkl', 'rb') as f:
        char_to_int = pickle.load(f)
    with open('models\int_to_char.pkl', 'rb') as f:
        int_to_char = pickle.load(f)
    with open('models\config.txt', 'r') as f:
        config = dict(line.split(': ', 1) for line in f.readlines())
        seq_length = int(config['seq_length'])
        vocab_size = int(config['vocab_size'])
    return model, char_to_int, int_to_char, seq_length, vocab_size

model, char_to_int, int_to_char, seq_length, vocab_size = load_model()

# Fixed generation function (from our earlier chat)
def generate_full_text(model, char_to_int, int_to_char, seed_text, gen_length=500, temperature=0.7):
    generated = seed_text
    space_char = ' '
    for _ in range(gen_length):
        input_seq = generated[-seq_length:]
        if len(input_seq) < seq_length:
            pad_len = seq_length - len(input_seq)
            input_seq = space_char * pad_len + input_seq
        x_pred = np.array([char_to_int.get(c, 0) for c in input_seq], dtype='int32').reshape(1, seq_length)
        preds = model.predict(x_pred, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_idx = np.random.choice(vocab_size, p=preds)
        generated += int_to_char[next_idx]
    return generated.strip()

# Streamlit UI
st.title("Shakespeare Text Generator")
st.write("Enter a seed phrase to generate Elizabethan-style prose or poetry!")

seed = st.text_input("Seed Text", value="to be or not to be", help="Start with a Shakespearean snippet for best results.")
temperature = st.slider("Temperature (Creativity)", 0.1, 1.5, 0.7, 0.1, help="Low: Predictable; High: Wild.")
gen_length = st.slider("Generation Length (Chars)", 50, 500, 300, 10)

if st.button("Generate!"):
    with st.spinner("Generating Text..."):
        output = generate_full_text(model, char_to_int, int_to_char, seed, gen_length, temperature)
        st.subheader("Generated Text:")
        st.write(output)

# Sidebar for info
with st.sidebar:
    st.info("**Model Info**\n- LSTM RNN trained on full Shakespeare corpus.\n- Vocab: 36 chars (a-z, punctuation).\n- Seq Length: 100.\n\n**Tips**\n- Try seeds like 'shall i compare thee'.\n- Refresh for new generations (stochastic).")
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 07:34:16 2022

Rebuilt by ChatGPT – same purpose as your original script but modified 
to avoid Keras/optree tree mapping issues by returning labels as a tensor,
and using temperature sampling in inference to help produce non-empty output.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from time import perf_counter
import numpy as np

# ============================
# 1) Hyperparameters
# ============================
# Originally set to a high number, but will update after vectorizer.adapt()
VOCAB_SIZE = 8192  
MAX_LENGTH = 40
EMBED_DIM = 256
LATENT_DIM = 512
NUM_HEADS = 8
BATCH_SIZE = 64

# ============================
# 2) Data Loading (External Files)
# ============================
def readTxtFile(strPath):
    with open(strPath, 'r', encoding="utf-8") as file:
        return file.read().replace("\n", " ")

def dsLoad(ins=498):
    refTexts, hSimTexts = [], []
    for i in range(1, ins):
        refTexts.append(readTxtFile(f"ArSummarizer/TSimAr/resources/Simplification_Datasets/References/{i}.txt"))
        hSimTexts.append(readTxtFile(f"ArSummarizer/TSimAr/resources/Simplification_Datasets/hSimplification/{i}.txt"))
    return refTexts, hSimTexts

refTexts, hSimTexts = dsLoad()
train_dataset_raw = tf.data.Dataset.from_tensor_slices((refTexts[:350], hSimTexts[:350]))
val_dataset_raw = tf.data.Dataset.from_tensor_slices((refTexts[350:], hSimTexts[350:]))

# ============================
# 3) Preprocessing & Tokenization
# ============================
def preprocess_text(sentence):
    sentence = tf.strings.lower(sentence)
    # Replace unwanted characters with a space (adjust regex as needed)
    sentence = tf.strings.regex_replace(sentence, r'([^\n\\u060C-\\u064A\.:؟?a-z0-9])', " ")
    sentence = tf.strings.strip(sentence)
    sentence = tf.strings.join(["[start]", sentence, "[end]"], separator=" ")
    return sentence

vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize=preprocess_text,
    output_mode="int",
    output_sequence_length=MAX_LENGTH,
)
all_texts = refTexts + hSimTexts
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(all_texts).batch(5))

# Update VOCAB_SIZE to the actual number of tokens in the vocabulary.
VOCAB = vectorizer.get_vocabulary()
VOCAB_SIZE = len(VOCAB)
print("Actual vocabulary size:", VOCAB_SIZE)

def vectorize_text(inputs, outputs):
    # Convert raw texts to integer sequences
    inputs = vectorizer(inputs)
    outputs = vectorizer(outputs)
    outputs = tf.pad(outputs, [[0, 1]])  # Pad outputs with one extra token

    # Cast to int32
    inputs = tf.cast(inputs, tf.int32)
    outputs = tf.cast(outputs, tf.int32)
    
    # Return features as a dict and labels as a tensor (not a dict)
    features = {
        "Encoder_Input_Embedding": inputs,
        "Decoder_Input_Embedding": outputs[:-1]
    }
    labels = outputs[1:]
    return features, labels

# Map the vectorization function.
train_dataset = (train_dataset_raw
                 .map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
                 .shuffle(20000)
                 .batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))
val_dataset = (val_dataset_raw
               .map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))

# Debug: Print dtypes of one batch to confirm they are numeric.
for features, labels in train_dataset.take(1):
    print("Encoder_Input_Embedding dtype:", features["Encoder_Input_Embedding"].dtype)
    print("Decoder_Input_Embedding dtype:", features["Decoder_Input_Embedding"].dtype)
    print("Labels dtype:", labels.dtype)

# ============================
# 4) FNet Encoder
# ============================
class FNetEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, **kwargs):
        super(FNetEncoder, self).__init__(**kwargs)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        inp_complex = tf.cast(inputs, tf.complex64)
        fft = tf.math.real(tf.signal.fft2d(inp_complex))
        proj_input = self.layernorm_1(inputs + fft)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

# ============================
# 5) Positional Embedding
# ============================
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

# ============================
# 6) FNet Decoder
# ============================
class FNetDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(FNetDecoder, self).__init__(**kwargs)
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(latent_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=out_1, value=encoder_outputs, key=encoder_outputs)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

# ============================
# 7) Build the Full Model
# ============================
def create_model():
    # Encoder
    encoder_inputs = keras.Input(shape=(None,), dtype="int32", name="Encoder_Input_Embedding")
    x_enc = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_inputs)
    encoder_outputs = FNetEncoder(EMBED_DIM, LATENT_DIM)(x_enc)
    
    # Decoder
    decoder_inputs = keras.Input(shape=(None,), dtype="int32", name="Decoder_Input_Embedding")
    x_dec = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_inputs)
    x_dec = FNetDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x_dec, encoder_outputs)
    x_dec = layers.Dropout(0.5)(x_dec)
    decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x_dec)
    
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="fnet")
    return model

# ============================
# 8) Train the Model
# ============================
fnet = create_model()
fnet.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fnet.fit(train_dataset, epochs=1, validation_data=val_dataset)
fnet.summary()

# ============================
# 9) Inference with Temperature Sampling
# ============================
def decode_sentence(input_sentence, temperature=1.0):
    tokenized_input = vectorizer(tf.constant("[start] " + input_sentence + " [end]"))
    tokenized_input = tf.cast(tokenized_input, tf.int32)
    start_idx = VOCAB.index("[start]")
    end_idx = VOCAB.index("[end]")
    tokenized_target = tf.expand_dims(start_idx, 0)
    decoded_sentence = ""

    for i in range(MAX_LENGTH):
        padded_target = tf.pad(tokenized_target, [[0, MAX_LENGTH - tf.shape(tokenized_target)[0]]])
        preds = fnet.predict({
            "Encoder_Input_Embedding": tf.expand_dims(tokenized_input, 0),
            "Decoder_Input_Embedding": tf.expand_dims(padded_target, 0)
        }, verbose=0)
        
        # Sample token with temperature
        logits = preds[0, i, :]
        next_token = tf.random.categorical(tf.expand_dims(logits / temperature, 0), num_samples=1).numpy()[0, 0]
        
        # Ensure next_token is within bounds
        if next_token >= len(VOCAB):
            print(f"Warning: Invalid token index {next_token}, skipping...")
            break
        
        if next_token == end_idx:
            break

        decoded_sentence += VOCAB[next_token] + " "
        tokenized_target = tf.concat([tokenized_target, [next_token]], axis=0)

    return decoded_sentence.strip()

# Test inference
print("Decoded sample:", decode_sentence("تعتبر جده البوابه الرئيسيه لمكه المكرمه، اقدس مدينه في الاسلام", temperature=1.0))

print("Done.")

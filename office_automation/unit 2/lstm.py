# 33_lstm_text_generator.py

import tensorflow as tf
import numpy as np

# --------------------------------------------------
# 1. Training text
# --------------------------------------------------
text = """
Artificial intelligence is a branch of computer science that focuses
on creating machines that can perform tasks that normally require
human intelligence. Artificial intelligence is used in many areas
such as healthcare, education, finance, robotics and transportation.

Machine learning is a part of artificial intelligence. Machine learning
allows computers to learn patterns from data without being explicitly
programmed. A machine learning model is trained using examples and
then uses the learned patterns to make predictions on new data.

Deep learning is a type of machine learning that uses artificial
neural networks with multiple layers. Deep learning is widely used
for image classification, speech recognition, natural language
processing and computer vision.

Natural language processing allows computers to understand and
generate human language. Text generation models learn patterns in
language and can generate new sentences based on previously observed
text.

Neural networks contain layers of interconnected neurons. During
training, the network adjusts its weights to reduce the difference
between its prediction and the expected output.

Recurrent neural networks are designed to process sequential data.
An RNN uses information from previous characters or words to predict
the next element in a sequence. This makes RNNs useful for text
generation and language modelling.

Long short term memory networks are an improved type of recurrent
neural network. LSTM networks contain gates that control the flow of
information through the network. They can remember important information
for a longer period of time and are useful for language generation.

Artificial intelligence continues to develop rapidly. Modern AI systems
can generate text, images, audio and computer programs. These systems
learn from large amounts of data and use neural networks to identify
complex patterns.
"""

text = text.lower()

# --------------------------------------------------
# 2. Create vocabulary
# --------------------------------------------------
vocab = sorted(set(text))

print("Number of unique characters:", len(vocab))
print("Vocabulary:", vocab)

# Character to integer
char_to_int = {
    char: i for i, char in enumerate(vocab)
}

# Integer to character
int_to_char = np.array(vocab)

# Encode complete text
encoded_text = np.array([
    char_to_int[char]
    for char in text
])

# --------------------------------------------------
# 3. Create training sequences
# --------------------------------------------------
sequence_length = 50

X = []
y = []

for i in range(
    len(encoded_text) - sequence_length
):

    # Input sequence
    X.append(
        encoded_text[i:i + sequence_length]
    )

    # Next character
    y.append(
        encoded_text[i + sequence_length]
    )

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------------------------------------
# 4. Build LSTM model
# --------------------------------------------------
model = tf.keras.Sequential([

    # Character embedding
    tf.keras.layers.Embedding(
        input_dim=len(vocab),
        output_dim=128
    ),

    # LSTM layer
    tf.keras.layers.LSTM(
        256,
        return_sequences=False
    ),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output layer
    tf.keras.layers.Dense(
        len(vocab),
        activation="softmax"
    )
])

model.summary()

# --------------------------------------------------
# 5. Compile model
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 6. Train model
# --------------------------------------------------
model.fit(
    X,
    y,
    epochs=100,
    batch_size=32
)

# --------------------------------------------------
# 7. Text generation function
# --------------------------------------------------
def generate_text(
    seed_text,
    num_characters=200,
    temperature=1.0
):

    generated_text = seed_text.lower()

    for _ in range(num_characters):

        # Take last sequence_length characters
        current_text = generated_text[
            -sequence_length:
        ]

        # Convert characters to integers
        encoded = [
            char_to_int.get(char, 0)
            for char in current_text
        ]

        # Pad if necessary
        if len(encoded) < sequence_length:

            encoded = (
                [0] *
                (sequence_length - len(encoded))
                + encoded
            )

        encoded = np.array(encoded).reshape(
            1, sequence_length
        )

        # Predict next character
        prediction = model.predict(
            encoded,
            verbose=0
        )[0]

        # Temperature controls randomness
        prediction = np.log(
            prediction + 1e-8
        ) / temperature

        prediction = np.exp(prediction)

        prediction = prediction / np.sum(
            prediction
        )

        # Randomly select character
        predicted_id = np.random.choice(
            len(vocab),
            p=prediction
        )

        predicted_char = int_to_char[
            predicted_id
        ]

        generated_text += predicted_char

    return generated_text


# --------------------------------------------------
# 8. Generate text
# --------------------------------------------------
seed = "artificial intelligence"

generated_text = generate_text(
    seed_text=seed,
    num_characters=200,
    temperature=0.7
)

print("\nGenerated Text:")
print(generated_text)
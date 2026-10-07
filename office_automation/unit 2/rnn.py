import tensorflow as tf
import numpy as np
text ="""
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
print("Text length:", len(text))
vocab = sorted(set(text))

print("Number of unique characters:", len(vocab))
print("Characters:", vocab)

char_to_int = {char: i for i, char in enumerate(vocab)}

int_to_char = np.array(vocab)

encoded_text = np.array([
    char_to_int[char] for char in text
])
sequence_length = 40

input_sequences = []
target_characters = []

for i in range(len(encoded_text) - sequence_length):
    input_sequences.append(
        encoded_text[i:i + sequence_length]
    )
    target_characters.append(
        encoded_text[i + sequence_length]
    )

X = np.array(input_sequences)
y = np.array(target_characters)

print("Input shape:", X.shape)
print("Target shape:", y.shape)

model = tf.keras.Sequential([
    
    tf.keras.layers.Embedding(
        input_dim=len(vocab),
        output_dim=64
    ),

    tf.keras.layers.SimpleRNN(128),

    # Predict next character
    tf.keras.layers.Dense(
        len(vocab),
        activation="softmax"
    )
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    X,
    y,
    epochs=100,
    batch_size=16
)

def generate_text(seed_text, num_characters=100):
    generated_text = seed_text.lower()
    for _ in range(num_characters):

        encoded = [
            char_to_int.get(char, 0)
            for char in generated_text[-sequence_length:]
        ]

        if len(encoded) < sequence_length:
            encoded = [0] * (
                sequence_length - len(encoded)
            ) + encoded

        encoded = np.array(encoded).reshape(
            1, sequence_length
        )
        prediction = model.predict(
            encoded,
            verbose=0
        )

        predicted_id = np.argmax(prediction[0])

        predicted_char = int_to_char[predicted_id]

        generated_text += predicted_char

    return generated_text

seed = "artificial intelligence"
generated = generate_text(
    seed,
    num_characters=100
)

print("\nGenerated Text:")
print(generated)
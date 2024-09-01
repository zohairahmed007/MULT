import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from transformers import TFBertModel, TFViTModel, BertTokenizer
sentiment_labels = ['Negative', 'Neutral', 'Positive']
class ViTLayer(tf.keras.layers.Layer):
    def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k'):
        super(ViTLayer, self).__init__()
        self.vit_model = TFViTModel.from_pretrained(vit_model_name)

    def call(self, inputs):
        # The ViT model expects input in (batch_size, channels, height, width)
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

        # Use the ViT model on the processed inputs
        outputs = self.vit_model(pixel_values=inputs).last_hidden_state
        return outputs[:, 0, :]  # Return the CLS token

def build_mult_model(max_length=128, img_size=(224, 224), num_classes=3):
    # Text input
    text_input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    text_attention_mask = Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
    
    # Load the BERT model
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    text_output = bert_model(text_input_ids, attention_mask=text_attention_mask)[0]
    text_cls_token = text_output[:, 0, :]  # CLS token

    # Image input
    image_input = Input(shape=(img_size[0], img_size[1], 3), dtype=tf.float32, name='image_input')
    
    # Apply ViT model through a custom Keras layer
    vit_layer = ViTLayer()
    image_cls_token = vit_layer(image_input)

    # Add classification layers for text and image models
    text_logits = layers.Dense(num_classes)(text_cls_token)
    image_logits = layers.Dense(num_classes)(image_cls_token)

    # Combine text and cross-modal features
    text_query = layers.Dense(768)(text_cls_token)
    image_key = layers.Dense(768)(image_cls_token)
    image_value = layers.Dense(768)(image_cls_token)

    attention_weights = layers.Attention()([text_query, image_key])
    cross_attention_output = layers.Add()([attention_weights, image_value])

    combined_features = layers.Concatenate()([text_cls_token, cross_attention_output])
    combined_features = layers.Dropout(0.3)(combined_features)
    combined_logits = layers.Dense(512, activation='relu')(combined_features)

    # Final classification layer for multimodal output
    multimodal_logits = layers.Dense(num_classes, activation='softmax')(combined_logits)

    # Build model with additional outputs for text and image logits
    model = Model(inputs=[text_input_ids, text_attention_mask, image_input], 
                  outputs=[multimodal_logits, text_logits, image_logits])

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Build the model
model = build_mult_model()

# Display the model architecture
model.summary()

# Load a sample text review
sample_review = "This is a great product. I really enjoyed using it and would recommend it to others."
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sample_text_tokens = tokenizer(sample_review, padding='max_length', max_length=128, return_tensors='tf')

# Load a sample image from the folder
image_path = '1.jpg'  # Update with your image path
image = tf.io.read_file(image_path)
sample_image_input = tf.image.decode_image(image, channels=3)
sample_image_input = tf.image.resize(sample_image_input, [224, 224])  # Resize the image
sample_image_input = tf.expand_dims(sample_image_input, axis=0)  # Add batch dimension

# Perform a forward pass with the sample inputs
multimodal_output, text_output, image_output = model([sample_text_tokens['input_ids'], 
                                                      sample_text_tokens['attention_mask'], 
                                                      sample_image_input])

# Apply softmax to the outputs to get probabilities
text_scores = tf.nn.softmax(text_output, axis=-1)
image_scores = tf.nn.softmax(image_output, axis=-1)
multimodal_scores = tf.nn.softmax(multimodal_output, axis=-1)

# Function to print sentiment scores and predicted labels
def print_sentiment_scores(scores, prefix):
    scores = scores.numpy()  # Convert to numpy array if needed
    predicted_labels = np.argmax(scores, axis=1)  # Get the index of the max score
    for i in range(len(scores)):
        print(f"{prefix} Sample {i+1}:")
        for j, label in enumerate(['Negative', 'Neutral', 'Positive']):
            print(f"{label}: {scores[i, j]:.4f}")
        # Ensure the predicted label index is valid
        if 0 <= predicted_labels[i] < len(sentiment_labels):
            print(f"Predicted Label: {sentiment_labels[predicted_labels[i]]}\n")
        else:
            print("Predicted Label: Invalid index\n")

# Print sentiment scores and predicted labels
print("Text Model Sentiment Scores:")
print_sentiment_scores(text_scores, "Text Model")

print("Image Model Sentiment Scores:")
print_sentiment_scores(image_scores, "Image Model")

print("Multimodal Model Sentiment Scores:")
print_sentiment_scores(multimodal_scores, "Multimodal Model")

##############################################################

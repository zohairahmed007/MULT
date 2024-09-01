import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import TFBertModel, TFViTModel, BertTokenizer
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Custom ViT Layer to extract CLS token and attention weights
class ViTLayer(tf.keras.layers.Layer):
    def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k'):
        super(ViTLayer, self).__init__()
        self.vit_model = TFViTModel.from_pretrained(vit_model_name, output_attentions=True)

    def call(self, inputs):
        # The ViT model expects input in (batch_size, channels, height, width)
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

        # Get the outputs and attention weights
        outputs = self.vit_model(pixel_values=inputs)
        cls_token_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        attention_weights = outputs.attentions[-1]  # Get the last attention weights

        return cls_token_output, attention_weights

# Function to build the multi-modal model
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
    image_cls_token, attention_weights = vit_layer(image_input)

    # Classification layers
    text_logits = layers.Dense(num_classes)(text_cls_token)
    image_logits = layers.Dense(num_classes)(image_cls_token)

    # Combine text and cross-modal features
    text_query = layers.Dense(768)(text_cls_token)
    image_key = layers.Dense(768)(image_cls_token)
    image_value = layers.Dense(768)(image_cls_token)

    attention_output = layers.Attention()([text_query, image_key])
    combined_features = layers.Concatenate()([text_cls_token, attention_output])
    combined_features = layers.Dropout(0.3)(combined_features)
    combined_logits = layers.Dense(512, activation='relu')(combined_features)

    # Final classification layer for multimodal output
    multimodal_logits = layers.Dense(num_classes, activation='softmax')(combined_logits)

    # Build model with the main output
    model = Model(inputs=[text_input_ids, text_attention_mask, image_input], 
                  outputs=multimodal_logits)

    return model
# Function to preprocess the text and images
def preprocess_text(text, tokenizer, max_length=128):
    return tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_tensors='tf')

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image

# Load and preprocess the MVSA dataset
df = pd.read_excel('output_dataframe.xlsx')

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df=df.head(50)
# Apply preprocessing to text and images
df['text_tokens'] = df['TextData'].apply(lambda x: preprocess_text(x, tokenizer))
df['image_input'] = df['ImagePath'].apply(preprocess_image)

# Convert labels to numerical format (assuming 'positive', 'neutral', 'negative' labels)
label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
df['label'] = df['image'].map(label_map)

    
# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)



def create_dataset(df, tokenizer, max_length=128, img_size=(224, 224), batch_size=16):
    # Check if 'text' and 'image_input' columns exist
    if 'text' not in df.columns or 'image_input' not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'image_input' columns.")

    # Preprocess the text data using the tokenizer
    text_inputs = df['text'].apply(lambda x: preprocess_text(x, tokenizer, max_length))
    
    # Extract the relevant parts from the tokenized output
    input_ids = np.array([text_input['input_ids'].numpy()[0] for text_input in text_inputs])
    attention_mask = np.array([text_input['attention_mask'].numpy()[0] for text_input in text_inputs])

    # Convert image_input and labels to numpy arrays
    image_input = np.stack(df['image_input'].values)
    labels = np.array(df['label'].values)
    
    # Ensure that all inputs have consistent shapes
    if input_ids.shape[1] != max_length or attention_mask.shape[1] != max_length:
        raise ValueError("Inconsistent sequence lengths detected. Ensure all sequences are padded to the same length.")
    
    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': tf.constant(input_ids, dtype=tf.int32),
            'attention_mask': tf.constant(attention_mask, dtype=tf.int32),
            'image_input': tf.constant(image_input, dtype=tf.float32)
        },
        tf.constant(labels, dtype=tf.int32)
    ))
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    return dataset



train_dataset = create_dataset(train_df,tokenizer)
val_dataset = create_dataset(val_df,tokenizer)
test_dataset = create_dataset(test_df,tokenizer)


for features, labels in train_dataset.take(1):  # Take 1 batch to inspect
    print("Features:")
    for key, value in features.items():
        print(f"  {key}: {value.shape}")
    print("Labels:", labels.shape)



# Define training parameters
batch_size = 16
epochs = 5

#train_dataset = train_dataset.shuffle(len(train_df)).batch(batch_size)
#val_dataset = val_dataset.batch(batch_size)
#test_dataset = test_dataset.batch(batch_size)



# Build the model
model = build_mult_model(num_classes=3)


#model.summary()
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f'Final Test Accuracy on MVSA Dataset: {test_accuracy:.4f}')

model.save('v3.h5')

from sklearn.metrics import classification_report

# Step 1: Make predictions
y_pred = model.predict(test_dataset)

# Step 2: Convert probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Step 3: Get true labels
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Step 4: Generate classification report
target_names = ['class_0', 'class_1', 'class_2']  # Adjust class names as needed
report = classification_report(y_true, y_pred_classes, target_names=target_names)

# Print the classification report
print(report)

# Optionally, print the final test accuracy
test_accuracy = np.mean(y_pred_classes == y_true)
print(f'Final Test Accuracy on MVSA Dataset: {test_accuracy:.4f}')











####################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Preprocess the image
# Load and preprocess the image
image_path = 'data/1.jpg'  # Replace with the path to your image
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)
image = tf.image.resize(image, [224, 224])
image = image / 255.0  # Normalize the image
img_array = np.expand_dims(image, axis=0)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example text input (replace with your actual text data)
text = "Your sample text here"

# Tokenize the text
text_inputs = tokenizer(text, return_tensors="tf", max_length=128, padding='max_length', truncation=True)

# Extract the individual text inputs
input_ids = text_inputs['input_ids']  # Shape [1, 128]
attention_mask = text_inputs['attention_mask']  # Shape [1, 128]

# Assuming img_array is a single preprocessed image with shape [224, 224, 3]
image_input = img_array  # Replace with your actual image preprocessing code

# Add a batch dimension to the image -> shape will be [1, 224, 224, 3]
image_input = tf.expand_dims(image_input, axis=0)

# Combine all inputs into a dictionary
inputs = {
    'input_ids': input_ids,           # Shape [1, 128]
    'attention_mask': attention_mask, # Shape [1, 128]
    'image_input': image_input        # Corrected shape [1, 224, 224, 3]
}

# Forward pass to get attention maps
attention_output = model(inputs)
attention_maps = attention_output[1]  # Extract attention map from the model's output

# Process the attention map (e.g., average over heads if applicable)
attention_map = tf.reduce_mean(attention_maps, axis=1)  # Average over heads (if applicable)

# Resize the attention map to match the image dimensions
attention_map = tf.image.resize(attention_map, (224, 224))  # Resize to match the image size
attention_map = tf.squeeze(attention_map)  # Remove unnecessary dimensions

# Function to superimpose the attention map on the original image
def superimpose_attention_map(image, attention_map, alpha=0.4, cmap='jet'):
    # Normalize the attention map
    attention_map = (attention_map - tf.reduce_min(attention_map)) / (tf.reduce_max(attention_map) - tf.reduce_min(attention_map))

    # Apply the colormap to the attention map
    attention_map_colored = plt.cm.get_cmap(cmap)(attention_map.numpy())
    attention_map_colored = tf.image.convert_image_dtype(attention_map_colored, dtype=tf.float32)

    # Superimpose the attention map on the original image
    superimposed_img = alpha * attention_map_colored[..., :3] + (1 - alpha) * image
    superimposed_img = tf.clip_by_value(superimposed_img, 0, 1)  # Ensure pixel values are within [0, 1]
    
    return superimposed_img

# Superimpose the attention map on the original image
superimposed_img = superimpose_attention_map(image_input[0], attention_map)

# Display the result
plt.imshow(superimposed_img)
plt.axis('off')

# Save the plot
output_path = 'superimposed_attention_map.png'  # Set the path where you want to save the image
#plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.show()


















############################################################
 #BERT
import tensorflow_hub as hub
path_text="./training_models/"

text_model = tf.keras.models.load_model(
       (path_text+'BERT_LSTM\\weights-improvement-10-0.76.hdf5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)


text_model.summary()
#print(places_mg_model.summary() )
for layer in text_model.layers:
  layer._name = 'text_'+layer.name

text_output = text_model.get_layer(name='text_dense_9').output
average_layer  = average([object_output, places_output,text_output])
multimodel_fusion_model = Model(inputs=[object_mg_model.input, places_mg_model.input,text_model.input], outputs=average_layer)

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def preprocess_text(sen):
    sentence = sen
    sentence = sentence.lower()
    return sentence

vec_preprocess_text = np.vectorize(preprocess_text)

encoder = LabelEncoder()
processed_test = vec_preprocess_text(new_df.textinput.values)


encoded_labels_train = encoder.fit_transform(new_df.label.values)
labels_test = utils.to_categorical(encoded_labels_train, 2)


print("Processed text sample:", processed_test[0])
print("Shape of train labels:", labels_test.shape)

import bert
import tensorflow_hub as hub
import tensorflow_text
# Import the BERT BASE model from Tensorflow HUB (layer, vocab_file and tokenizer)
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

# Preprocessing of texts according to BERT
def get_masks(text, max_length):
    """Mask for padding"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]

    return np.asarray([1]*len(tokens) + [0] * (max_length - len(tokens)))
vec_get_masks = np.vectorize(get_masks, signature = '(),()->(n)')

def get_segments(text, max_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]
    
    segments = []
    current_segment_id = 0
    with_tags = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.asarray(segments + [0] * (max_length - len(tokens)))
vec_get_segments = np.vectorize(get_segments, signature = '(),()->(n)')

def get_ids(text, tokenizer, max_length):
    """Token ids from Tokenizer vocab"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = np.asarray(token_ids + [0] * (max_length-length))
    return input_ids
vec_get_ids = np.vectorize(get_ids, signature = '(),(),()->(n)')


def prepare(text_array, tokenizer, max_length = 128):
    
    ids = vec_get_ids(text_array, 
                      tokenizer, 
                      max_length).squeeze()
    masks = vec_get_masks(text_array,
                      max_length).squeeze()
    segments = vec_get_segments(text_array,
                      max_length).squeeze()

    return ids, segments, masks
     

max_length = 40 # that must be set according to your dataset
ids_train, segments_train, masks_train = prepare(processed_test,
                                                 tokenizer,
                                                 max_length)
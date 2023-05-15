import numpy as np
import re
import nltk
# import tensorflow_hub as hub
from keras import Input, Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, Flatten
from matching_items_code.product_matching.entities import import_data, Parameters


class PreprocessData:
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.frame = import_data()

    def group_strings(self) -> None:
        self.frame['full_text'] = self.frame.name + ' ' + self.frame.item_description
        self.frame.drop(['item_description'], axis=1, inplace=True)

    def _clean_text(self, text: str) -> object:
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        text = ' '.join([word for word in text.split() if word not in self.stop_words])
        return text

    def clean_full_text(self) -> None:
        self.frame['cleaned_text'] = self.frame['full_text'].apply(self._clean_text)
        self.frame.drop(['full_text'], axis=1, inplace=True)
        return self.frame



# Tokenize and pad the sequences
prep = PreprocessData()
prep.group_strings()
df = prep.clean_full_text()


tokenizer = Tokenizer(num_words=Parameters.vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
padded_sequences = pad_sequences(sequences, maxlen=Parameters.max_len, padding='post', truncating='post')


model = Sequential([
    Embedding(Parameters.vocab_size, Parameters.embedding_dim,
              input_length=Parameters.max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(padded_sequences, padded_sequences, epochs=10, batch_size=32)

# Evaluate the model and threshold for matching products
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Create embeddings for the test samples
test_sequences = tokenizer.texts_to_sequences(test_data['cleaned_text'])
test_padded_sequences = pad_sequences(test_sequences, maxlen=Parameters.max_len, padding='post', truncating='post')
test_embeddings = model.predict(test_padded_sequences)

# Compute cosine similarity between all the test samples
similarity_matrix = cosine_similarity(test_embeddings)

# Set a similarity threshold to decide if two products are the same (e.g., 0.8)
similarity_threshold = 0.8

# Apply the threshold to the similarity_matrix and compare with the ground truth
# Assuming test_data has a boolean column named 'same_product' which is True if the
# products are the same and False otherwise
predicted_same_product = similarity_matrix >= similarity_threshold
true_same_product = np.array(test_data['same_product'])

# Calculate the model's performance metrics
accuracy = np.sum(predicted_same_product == true_same_product) / len(true_same_product)
print(f'Accuracy: {accuracy}')

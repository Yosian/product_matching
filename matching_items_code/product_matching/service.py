from keras.losses import cosine_similarity
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

from matching_items_code.product_matching.entities import Parameters
from matching_items_code.product_matching.repository import PreprocessData, GenerateModel

# Tokenise and pad the sequences
prep = PreprocessData()
prep.group_strings()
df = prep.clean_full_text()

# Generate model engine
gen = GenerateModel(frame=df)
model, tokeniser = gen.generate()

# Evaluate the model and threshold for matching products
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Create embeddings for the test samples
test_sequences = tokeniser.texts_to_sequences(test_data['cleaned_text'])
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
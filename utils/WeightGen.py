from utils.DataProviderLight import DataProviderLight
from preprocessing.Word2Vec import Word2Vec
# generate weight matrix and save to a file for later reuse

SAMPLE_SIZE = 100000
BATCH_SIZE = 256
EPOCHS = 5
FEATURES = 500
# change device to "cpu" if cuda not available
DEVICE = "cuda"

# raw words and labels
file_words = open("../data/unique_equations.json")
file_labels = open("../data/unique_labels.json")
data_provider = DataProviderLight(file_words, file_labels, sample_size=SAMPLE_SIZE)
file_matrix = open("../tests/matrix8.json", "w")

word2vec = Word2Vec(data_provider, FEATURES, DEVICE)
word2vec.train(EPOCHS, BATCH_SIZE)
word2vec.save_weights(file_matrix)
print("done")

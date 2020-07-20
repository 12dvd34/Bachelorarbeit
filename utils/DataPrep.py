import json
from utils.DataProvider import DataProvider
# some preprocessing on the data


# remove duplicates
def clean_data():
    file_tokens = open("../data/all_eq_tex_tokenized.json")
    file_labels = open("../data/all_eq_tex_labels.json")
    data_provider = DataProvider()
    data_provider.load_tokens(file_tokens)
    data_provider.load_labels(file_labels)
    unique_words = []
    unique_labels = []
    for index in range(len(data_provider.words)):
        if index % 1000 == 0:
            print(index)
        if data_provider.words[index] not in unique_words:
            unique_words.append(data_provider.words[index])
            unique_labels.append(data_provider.labels[index])
    print(len(unique_words))
    print(len(unique_labels))
    words_out = open("../data/unique_equations.json", "w")
    words_out.write(json.dumps(unique_words))
    words_out.close()
    labels_out = open("../data/unique_labels.json", "w")
    labels_out.write(json.dumps(unique_labels))
    labels_out.close()


# map equations into n-dimensional space for faster loading
def preprocess_data():
    file_tokens = open("../data/unique_equations.json")
    file_weights = open("../tests/matrix8.json")
    data_provider = DataProvider()
    data_provider.load_tokens(file_tokens)
    data_provider.load_weights(file_weights)
    data_provider.generate_token_mapping()
    word_vectors = []
    for index, word in enumerate(data_provider.words[:100000]):
        if index % 1000 == 0:
            print(str(index) + "/" + str(len(data_provider.words)))
        word_vectors.append(data_provider.generate_word_vector(word).tolist())
    file_vectors = open("../tests/embedding8.json", "w")
    file_vectors.write(json.dumps(word_vectors))
    file_vectors.close()


preprocess_data()


from utils.DataProviderLight import DataProviderLight
import matplotlib.pyplot as plt
# analyze some aspects of the data

file_words = open("../data/unique_equations.json")
file_labels = open("../data/unique_labels.json")
data_provider = DataProviderLight(file_words, file_labels)


def count_word_lengths():
    lengths = {}
    for word in data_provider.words:
        if len(word) in lengths.keys():
            lengths[len(word)] = lengths[len(word)] + 1
        else:
            lengths[len(word)] = 1
    x = [key for key in lengths.keys()]
    y = [lengths[key] for key in lengths.keys()]
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="tab:blue", s=10)
    ax.set_xlabel("Länge der Formel")
    ax.set_ylabel("Anzahl der Formeln")
    plt.show()
    print("done")


def count_class_sizes():
    labels = {}
    for label in data_provider.labels:
        if label in labels:
            labels[label] = labels[label] + 1
        else:
            labels[label] = 1
    label_counts = {}
    for label in labels:
        if labels[label] in label_counts:
            label_counts[labels[label]] = label_counts[labels[label]] + 1
        else:
            label_counts[labels[label]] = 1
    x = [key for key in label_counts.keys()]
    y = [label_counts[key] for key in label_counts.keys()]
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="tab:blue", s=10)
    ax.set_xlabel("Größe der Klasse")
    ax.set_ylabel("Anzahl der Klassen")
    plt.show()
    print("done")


count_class_sizes()

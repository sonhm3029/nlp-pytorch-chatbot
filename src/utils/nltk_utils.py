import nltk
import underthesea
import numpy as np
nltk.download('punkt')


def tokenize(sentence):
    return underthesea.word_tokenize(sentence)


def bag_of_words(tokenized_sentence, all_worlds):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [0,     1,      0,      1,      0,      0,      0]
    """
    bag = np.zeros(len(all_worlds), dtype=np.float32)
    for idx, w in enumerate(all_worlds):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag



import collections
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.snowball import EnglishStemmer


def plot_scatter(x, y, title):
    plt.plot(x, y)
    plt.xlabel("log(rank)")
    plt.ylabel("log(count)")
    plt.title(title)
    plt.show()


def plot_word_occurrences(_tokens, title):
    count = collections.Counter(_tokens).most_common()
    counts = np.array([i[1] for i in count]).astype(np.float64)
    y = np.array([np.log(i) for i in counts])
    x = np.log(np.arange(1, len(count) + 1))
    plot_scatter(x, y, title)
    print([i[0] for i in count[:20]])


def b(_tokens):
    plot_word_occurrences(_tokens, "Complete Token Occurrences")


def c(_tokens):
    stops = nltk.corpus.stopwords.words('english')
    filtered = [token for token in _tokens if token not in stops]
    plot_word_occurrences(filtered, "No Stop-Words Token Occurrences")


def d(_tokens):
    stemmer = EnglishStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in _tokens]
    plot_word_occurrences(stemmed_tokens, "Stemmed Token Occurrences")


def e(tokens):
    pos_tagging = nltk.pos_tag(tokens)
    phrases = []
    tag = 0
    is_adj = False
    while tag < len(pos_tagging):
        if pos_tagging[tag][1][0] == 'N' and len(pos_tagging[tag][0]) > 1:  # find noun
            phrase = pos_tagging[tag][0]  # add first noun
            k = tag - 1
            while k >= 0 and pos_tagging[k][1][0] == 'J':  # Go backwards search adjectives
                is_adj = True
                phrase = pos_tagging[k][0] + " " + phrase
                k -= 1
            k = tag + 1  # Add additional nouns if exist/
            while k < len(pos_tagging) and pos_tagging[k][1][0] == 'N':
                phrase = phrase + " " + pos_tagging[k][0]
                k += 1
            if is_adj:  # Add phrase only if there is adjective\s + noun\s
                phrases.append(phrase)
            tag = k
            is_adj = False
        tag += 1
    plot_word_occurrences(phrases, "POS-tagging Token Occurrences")


def main():
    book = open("book.txt", 'r', encoding="utf8").read()
    _tokens = nltk.RegexpTokenizer(r'\w+').tokenize(book)
    e(_tokens)


if __name__ == "__main__":
    main()

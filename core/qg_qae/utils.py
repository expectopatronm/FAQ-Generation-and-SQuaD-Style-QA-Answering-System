import re

import benepar

import nltk
from nltk import sent_tokenize


benepar.download("benepar_en3")
benepar_parser = benepar.Parser("benepar_en3")


nltk.download("punkt")


def remove_duplicates(questions):
    """Description."""

    return list(set(questions))


def preprocess_corpus(corpus):
    """Description."""

    corpus_final = []
    corpus = corpus.split("\n\n")[0].split("\r\n\r\n")
    for passage in corpus:
        corpus_final.append(passage.replace("\r\n", " "))
    return corpus_final


def obtain_phrase_to_sentence_position(sentence, phrases):
    """Description."""

    visited_dict = {}
    for phrase in phrases:
        match = [m.start(0) for m in re.finditer(phrase, sentence)]
        if match:
            if phrase in visited_dict:
                for value in visited_dict[phrase]:
                    match.remove(value)
                visited_dict[phrase].append([match][0][0])
            else:
                visited_dict[phrase] = [[match][0][0]]

    phrase_position_dict = {}
    for phrase in phrases:
        phrase_position_dict[visited_dict[phrase][0]] = phrase
        visited_dict[phrase].remove(visited_dict[phrase][0])

    return phrase_position_dict


def extract_phrase(sentence, label):
    """
    'NP' are phrases where a noun acts as the head word.
    Noun phrases act as a subject or object to a verb.

    'VP' phrases are lexical units that have a verb acting as the head word.
    Usually, there are two forms of verb phrases.
    One form has the verb components as well as other entities such as nouns,
    adjectives, or adverbs as parts of the object.
    """

    phrases = []
    trees = benepar_parser.parse_sents(sent_tokenize(sentence))
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == label:
                t = subtree
                t = " ".join(t.leaves())
                phrases.append(t)
    return phrases

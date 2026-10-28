#!/usr/bin/env python3
import math
import os
import re
from collections import Counter, defaultdict

DOC_LIST_FILE = "tfidf_docs.txt"
STOPWORDS_FILE = "stopwords.txt"


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_stopwords(path):
    if not os.path.exists(path):
        return set()
    text = read_file(path)
    return {word.strip().lower() for word in text.split() if word.strip()}


def clean_document(text):
    # Remove website links first
    text = re.sub(r'https?://\S+', ' ', text)
    # Keep only word characters and whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    # Lowercase
    text = text.lower()
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text, stopwords):
    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words)


def stem_word(word):
    if word.endswith("ing") and len(word) > 3:
        word = word[:-3]
    if word.endswith("ly") and len(word) > 2:
        word = word[:-2]
    if word.endswith("ment") and len(word) > 4:
        word = word[:-4]
    return word


def stem_and_lemmatize(text):
    words = [stem_word(w) for w in text.split()]
    return " ".join(words)


def preprocess_document(text, stopwords):
    cleaned = clean_document(text)
    no_stopwords = remove_stopwords(cleaned, stopwords)
    reduced = stem_and_lemmatize(no_stopwords)
    reduced = re.sub(r'\s+', ' ', reduced).strip()
    return reduced


def preprocess_all(doc_files, stopwords):
    preprocessed = {}
    for doc_file in doc_files:
        original_text = read_file(doc_file)
        processed_text = preprocess_document(original_text, stopwords)
        preprocessed[doc_file] = processed_text

        output_name = f"preproc_{os.path.basename(doc_file)}"
        write_file(output_name, processed_text)
    return preprocessed


def compute_tf(preprocessed_docs):
    tf_data = {}
    freq_data = {}
    for doc_file, text in preprocessed_docs.items():
        words = text.split()
        counts = Counter(words)
        total_terms = len(words)
        freq_data[doc_file] = counts

        tf_scores = {}
        for word, count in counts.items():
            tf_scores[word] = count / total_terms if total_terms else 0
        tf_data[doc_file] = tf_scores
    return freq_data, tf_data


def compute_idf(preprocessed_docs):
    total_docs = len(preprocessed_docs)
    doc_freq = defaultdict(int)

    for text in preprocessed_docs.values():
        unique_words = set(text.split())
        for word in unique_words:
            doc_freq[word] += 1

    idf_scores = {}
    for word, count in doc_freq.items():
        idf_scores[word] = math.log(total_docs / count) + 1
    return idf_scores


def compute_tfidf(tf_data, idf_scores):
    tfidf_data = {}
    for doc_file, tf_scores in tf_data.items():
        scores = {}
        for word, tf_value in tf_scores.items():
            scores[word] = round(tf_value * idf_scores[word], 2)
        tfidf_data[doc_file] = scores
    return tfidf_data


def top_five_words(tfidf_scores):
    top_data = {}
    for doc_file, scores in tfidf_scores.items():
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_data[doc_file] = ranked[:5]
    return top_data


def write_tfidf_outputs(top_scores):
    for doc_file, items in top_scores.items():
        output_name = f"tfidf_{os.path.basename(doc_file)}"
        write_file(output_name, str(items))


def load_doc_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    if not os.path.exists(DOC_LIST_FILE):
        raise FileNotFoundError(f"Missing required file: {DOC_LIST_FILE}")

    doc_files = load_doc_list(DOC_LIST_FILE)
    stopwords = load_stopwords(STOPWORDS_FILE)

    preprocessed_docs = preprocess_all(doc_files, stopwords)
    _, tf_data = compute_tf(preprocessed_docs)
    idf_scores = compute_idf(preprocessed_docs)
    tfidf_scores = compute_tfidf(tf_data, idf_scores)
    top_scores = top_five_words(tfidf_scores)
    write_tfidf_outputs(top_scores)


if __name__ == "__main__":
    main()

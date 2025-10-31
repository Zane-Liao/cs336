import os
# import kenlm
import mmh3
import hashlib
import fasttext
# import bitarray
import requests
import shutil
import regex as re
from io import BytesIO
from typing import Any
import numpy as np
from nltk import word_tokenize
from dataclasses import dataclass
from itertools import combinations
from collections import Counter, defaultdict
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

EMAIL = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE = re.compile(r"(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
                  r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b")

R_EMAIL = "|||EMAIL_ADDRESS|||"
R_PHONE = "|||PHONE_NUMBER|||"
R_IPV4 = "|||IP_ADDRESS|||"

MIN_LEN = 50
MAX_LEN = 100_000
MIN_MEAN_LEN = 3
MAX_MEAN_LEN = 10
LINE_O = 0.3
LINE_L = 0.8

# Problem (extract_text): 3 points
def html_trans_text(html_bytes: bytes) -> str | None:
    """
        TEST:
            Implement the adapter [run_extract_text_from_html_bytes] and make sure it passes
            uv run pytest -k test_extract_text_from_html_bytes
    """
    data = detect_encoding(html_bytes)
    
    html_str = html_bytes.decode(data, 'ignore')
    
    text = extract_plain_text(html_str)
    
    return text


# Problem (language_identification): 6 points
def language_ident(unicode_str: str) -> tuple[Any, float]:
    """
        TEST:
            Implement the adapter [run_identify_language] and
            make sure it passes both tests in
            uv run pytest -k test_identify_language
    """
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_path = "var/lid.176.bin"
    download_file(model_url, model_path)
    
    model = fasttext.load_model(model_path)
    predicted, scores = [], []
    
    # Split Line
    lines = [line.strip() for line in unicode_str.split("\n") if line.strip()]
    
    for line in lines:
        labels, score = model.predict(line)
        predicted.append(labels[0])
        scores.append(score[0])
    
    # .most_common(1) = [('en', 3)]
    # .most_common(1)[0] = ('en', 3)
    # .most_common(1)[0][0] = 'en'
    common_labels = Counter(predicted).most_common(1)[0][0]
    mean_score = float(np.mean(score))
    
    predicted_language = common_labels.replace('__label__', '')
    
    return (predicted_language, mean_score)


# Problem (mask_pii): 3 points
def email_trans_string(email_str: str) -> tuple[str, int]:
    """
        TEST:
            Implement the adapter [run_mask_emails] and make sure it passes all tests in
            uv run pytest -k test_mask_emails
    """    
    masked_text, r_num = EMAIL.subn(R_EMAIL, email_str)
    
    return (masked_text, r_num)


def phone_trans_string(phone_str: str) -> tuple[str, int]:
    """
        TEST:
            Implement the adapter [run_mask_phone_numbers] and make sure it passes
            uv run pytest -k test_mask_phones
    """
    masked_text, r_num = PHONE.subn(R_PHONE, phone_str)

    return (masked_text, r_num)


def ipv4_trans_string(ipv4_str: str) -> tuple[str, int]:
    """
        TEST:
            Implement the adapter [run_mask_ips] and make sure it passes
            uv run pytest -k test_mask_ips
    """
    masked_text, r_num = IPV4.subn(R_IPV4, ipv4_str)
    
    return (masked_text, r_num)


# Problem (harmful_content): 6 points
def check_nsfw(text: str) -> tuple[Any, float]:
    """
        TEST:
            Implement the adapter [run_classify_nsfw] and make sure it passes
            uv run pytest -k test_classify_nsfw
    """
    # jigsaw_fasttext_bigrams_nsfw_final.bin rename nsfw.bin
    model_path = "var/nsfw.bin"
    model = fasttext.load_model(model_path)
    predicted, scores = [], []
    
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    for line in lines:
        labels, score = model.predict(line)
        predicted.append(labels[0])
        scores.append(score[0])
    
    common_labels = Counter(predicted).most_common(1)[0][0]
    mean_score = float(np.mean(score))
    
    nsfw_language = common_labels.replace('__label__', '')
    
    return (nsfw_language, mean_score)


def check_oxic_speech(text: str) -> tuple[Any, float]:
    """
        TEST:
            Implement the adapter [run_classify_toxic_speech] and make sure it passes
            uv run pytest -k test_classify_toxic_speech
    """
    # jigsaw_fasttext_bigrams_hatespeech_final.bin rename hatespeech.bin
    model_path = "var/hatespeech.bin"
    model = fasttext.load_model(model_path)
    predicted, scores = [], []
    
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    for line in lines:
        labels, score = model.predict(line)
        predicted.append(labels[0])
        scores.append(score[0])
    
    common_labels = Counter(predicted).most_common(1)[0][0]
    mean_score = float(np.mean(score))
    
    hate_speech_language = common_labels.replace('__label__', '')
    
    return (hate_speech_language, mean_score)
    

# Problem (gopher_quality_filters): 3 points
def gopher_filter(text: str) -> bool:
    """
        TEST:
            Implement the adapter [run_gopher_quality_filter]. Then,
            make sure your filters pass the tests in
            uv run pytest -k test_gopher
    """
    if not text or not text.strip():
        return False
    
    words = word_tokenize(text)
    n_words = len(words)
    
    if n_words < MIN_LEN or n_words > MAX_LEN:
        return False
    
    mean_len = sum(len(n) for n in words) / n_words
    if mean_len < MIN_MEAN_LEN or mean_len > MAX_MEAN_LEN:
        return False

    lines = text.splitlines()
    if lines:
        e_split = sum(1 for line in lines if line.strip().endswith("..."))
        if (e_split / len(lines)) > LINE_O:
            return False
        
    alphabetic_words = sum(1 for w in words if any(ch.isalpha() for ch in w))
    if alphabetic_words / n_words < LINE_L:
        return False
    
    return True    


# Problem (quality_classifier): 15 points
def quality_classifier(text: str) -> tuple[Any, float]:
    """
        TEST:
            Implement the adapter [run_classify_quality]. As a sanity check, make sure
            it correctly classifies the two examples we provide by running
            uv run pytest -k test_classify_quality
    """
    quality_model = fasttext.load_model("quality_classifier.bin")
    
    labels, scores = quality_model.predict(text.replace("\n", " "))
    
    label = labels[0].replace("__label__", "")
    score = float(scores[0])
    
    return (label, score)


def model_classifier_label():
    model = fasttext.train_supervised(
        input="train.txt",
        lr=0.1,
        epoch=10,
        wordNgrams=2,
        dim=100,
        loss="softamx",
    )
    
    model.save_model("quality_classifier.bin")


# Problem (exact_deduplication): 3 points
def line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
        TEST:
            Implement the adapter [run_exact_line_deduplication] and make sure it passes
            uv run pytest -k test_exact_line_deduplication
    """
    line_count = defaultdict(int)
    
    for path in input_files:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                h = hashlib.sha512(line.encode('utf-8')).hexdigest()
                line_count[h] += 1
    
    os.makedirs(output_directory, exist_ok=True)
    
    for path in input_files:
        rename = os.path.basename(path)
        output_path = os.path.join(output_directory, rename)
        with open(path, 'r', encoding='utf-8') as fr, open(output_path, 'w', encoding='utf-8') as fw:
            for line in fr:
                line = line.rstrip()
                h = hashlib.sha512(line.encode('utf-8')).hexdigest()
                if line_count[h] == 1:
                    fw.write(line+'\n')


# Problem (minhash_deduplication): 8 points
def minhash_lsh_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
        TEST:
            Implement the adapter [run_minhash_deduplication] and make sure it passes
            uv run pytest -k test_minhash_deduplication
    """
    os.makedirs(output_directory, exist_ok=True)
    
    docs = []
    for path in input_files:
        with open(path, "r", encoding='utf-8') as f:
            content = f.read()
            norm_text = normalize_text(content)
            ngrams_set = get_ngrams(norm_text, ngrams)
            signature = compute_minhash(ngrams_set, num_hashes)
            docs.append({
                "path": path,
                "content": content,
                "norm": norm_text,
                "ngrams": ngrams_set,
                "sig": signature,
                })
    
    # Detect duplicate text
    unique_docs = []
    seen_texts = set()
    for doc in docs:
        if doc["content"] not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(doc["content"])
    
    # LSH bucketing
    band_buckets = defaultdict(list)
    for idx, doc in enumerate(unique_docs):
        for band_hash in lsh_buckets(doc["sig"], num_bands):
            band_buckets[band_hash].append(idx)
            
    # Identify fuzzy duplicates
    to_remove = set()
    for bucket_docs in band_buckets.values():
        for i, j in combinations(bucket_docs, 2):
            if i in to_remove or j in to_remove:
                continue
            
            sim = compute_jaccard(unique_docs[i]["ngrams"], unique_docs[j]["ngrams"])
            if sim >= jaccard_threshold:
                to_remove.add(j)
                
    kept = [doc for i, doc in enumerate(unique_docs) if i not in to_remove]
    
    # Write deduplicated docs
    for doc in kept:
        name = os.path.basename(doc["path"])
        output_path = os.path.join(output_directory, name)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc["content"])


def normalize_text(text: str) -> str:
    """
        Normalize whitespace and lowercase text
        
        Regex Simple Example:
            >>> text = "hello   world\tthis is\npython"
            >>> re.split(r"\\s+", text)
            ['hello', 'world', 'this', 'is', 'python']
    """
    text = text.replace("\r\n", "\n").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_ngrams(text: str, n: int) -> set[str]:
    """
        Split text into n-grams
        
        Example:
            >>> text = "abcd"
            >>> n = 2
            >>> result = {text[i:i + n] for i in range(max(0, len(text) - n + 1))}
            >>> result
            {'ab', 'bc', 'cd'}
    """
    return { text[i:i+n] for i in range(max(0, len(text) - n + 1)) }

def compute_minhash(ngrams: set[str], num_hashes: int) -> list[int]:
    """Compute MinHash"""
    signature = []
    for seed in range(num_hashes):
        min_hash = min(mmh3.hash(gram, seed, signed=False) for gram in ngrams)
        signature.append(min_hash)

    return signature

def lsh_buckets(signature: set[int], num_bands: int) -> list[int]:
    """Divide signature into bands and compute hash for each band"""
    rows_band = len(signature) // num_bands
    buckets = []
    for i in range(num_bands):
        start = 1 * rows_band
        end = (i + 1) * rows_band
        # ...
        band_hash = hash(tuple(signature[start:end]))
        buckets.append(band_hash)
    
    return buckets
    
##################################################################################################
##################################################################################################

def download_file(url: str, filename: str):
    """Download `url` and save the contents to `filename`.  Skip if `filename` already exists."""
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
        }
        response = requests.get(url, headers=headers)
        with open(filename, "wb") as f:
            shutil.copyfileobj(BytesIO(response.content), f)
            
def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union
import os
import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
# import kenlm
import fasttext
import itertools
import mmh3
import resiliparse
# import bitarray
from dataclasses import dataclass
from nltk import word_tokenize
from typing import Any
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from collections import Counter
from io import BytesIO
import requests
import os
import shutil
import regex as re

EMAIL = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE = re.compile(r"(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
                  r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b")

R_EMAIL = "|||EMAIL_ADDRESS|||"
R_PHONE = "|||PHONE_NUMBER|||"
R_IPV4 = "|||IP_ADDRESS|||"


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
def gopher_filter(text: str) -> tuple[Any, float]:
    """
        TEST:
            Implement the adapter [run_gopher_quality_filter]. Then,
            make sure your filters pass the tests in
            uv run pytest -k test_gopher
    """
    raise NotImplementedError


# Problem (quality_classifier): 15 points
def quality_classifier(text: str) -> bool:
    """
        TEST:
            Implement the adapter [run_classify_quality]. As a sanity check, make sure
            it correctly classifies the two examples we provide by running
            uv run pytest -k test_classify_quality
    """
    raise NotImplementedError


def classifier_label():
    raise NotImplementedError


# Problem (exact_deduplication): 3 points
def line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
        TEST:
            Implement the adapter [run_exact_line_deduplication] and make sure it passes
            uv run pytest -k test_exact_line_deduplication
    """
    raise NotImplementedError


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
    raise NotImplementedError

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
            

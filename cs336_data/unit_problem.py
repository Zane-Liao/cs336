from fastwarc.warc import ArchiveIterator, WarcRecordType
from fastwarc.stream_io import FileStream
from data_process import *


def warc_text(max_print: int):
    """
        Question: Run your text extraction function on a single WARC file. Compare
        its output to the extracted text in the corresponding WET file.
        What differences and/or similarities do you notice? Which extraction seems
        better?
    """
    count = 0
    with FileStream("../data/example.warc.gz") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response:
                raw_bytes = record.reader.read()
                text = html_trans_text(raw_bytes)

                if count < max_print:
                    count += 1
                    print(text[200:])


def extract_text_from_warc(warc_path: str):
    texts = []
    with FileStream(warc_path) as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == "response":
                try:
                    html = record.reader.read()
                    if html:
                        text = extract_plain_text(html)
                        texts.append(text.replace("\n", " "))
                except Exception as e:
                    print(f"[WARN] {e}")
    print(f"[INFO] Extracted {len(texts)} texts from {warc_path}")
    return texts

def inspect_warc(path):
    from collections import Counter
    counts = Counter()
    with FileStream(path) as stream:
        for record in ArchiveIterator(stream):
            counts[record.record_type] += 1
    print(counts)


def file_open_text():
    pos_texts = extract_text_from_warc("../var/positive_samples.warc.gz")
    neg_texts = extract_text_from_warc("../var/negative_samples.warc.gz")
    
    pos_texts = [t for t in pos_texts if gopher_filter(t)]
    neg_texts = [t for t in neg_texts if gopher_filter(t)]
    
    with open("train.txt", "w") as f:
        for text in pos_texts:
            f.write(f"__label__high {text}\n")
        for text in neg_texts:
            f.write(f"__label__low {text}\n")


def all_main():
    # warc_text(max_print=5)
    file_open_text()
    inspect_warc("../var/positive_samples.warc.gz")
    inspect_warc("../var/negative_samples.warc.gz")


if __name__ == '__main__':
    all_main()
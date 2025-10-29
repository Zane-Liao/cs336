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


def all_main():
    warc_text(max_print=5)


if __name__ == '__main__':
    all_main()
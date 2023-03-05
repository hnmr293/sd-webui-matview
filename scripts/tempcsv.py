from contextlib import contextmanager
import tempfile
import csv

@contextmanager
def csv_write(close: bool = True):
    io = tempfile.NamedTemporaryFile(mode='w')
    w = csv.writer(io)
    try:
        yield w, io
    finally:
        if close:
            io.close()

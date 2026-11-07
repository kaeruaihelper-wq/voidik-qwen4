import sys
from types import SimpleNamespace

from scripts.prepare_data import tokenize_split


class FakeDataset:
    def __init__(self, documents):
        self.documents = documents

    def iter(self, batch_size):
        del batch_size
        yield {"text": self.documents}

    def skip(self, count):
        return FakeDataset(self.documents[count:])


class FakeTokenizer:
    def get_vocab_size(self):
        return 128

    def encode_batch(self, texts):
        return [SimpleNamespace(ids=[ord(text[0])]) for text in texts]


def test_validation_budget_skips_only_documents_written(tmp_path, monkeypatch):
    documents = [f"doc-{index}" for index in range(20)]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: FakeDataset(documents)),
    )
    tokenizer = FakeTokenizer()

    validation = tokenize_split(
        "text",
        "train",
        "text",
        tokenizer,
        tmp_path,
        max_tokens=4,
        shard_tokens=100,
        eos_id=127,
        data_files="unused",
        min_chars=1,
        max_docs=None,
        prefix="val",
    )
    training = tokenize_split(
        "text",
        "train",
        "text",
        tokenizer,
        tmp_path,
        max_tokens=10,
        shard_tokens=100,
        eos_id=127,
        data_files="unused",
        min_chars=1,
        max_docs=None,
        prefix="train",
        skip_docs=validation["docs"],
    )

    assert validation["tokens"] == 4
    assert validation["docs"] == 2
    assert training["tokens"] == 10
    assert training["docs"] == 5
    assert training["shards"]

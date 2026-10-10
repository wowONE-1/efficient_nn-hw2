import os
import regex as re
from abc import ABC
from collections.abc import Iterator, Iterable
import multiprocessing as mp
from tqdm.auto import tqdm


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge(indices: tuple[int], new_pair: tuple[int, int], new_ind: int) -> list[int]:
    new_indices = []
    i = 0
    while i < len(indices):
        if (i + 1 < len(indices)) and (indices[i] == new_pair[0]) and (indices[i+1] == new_pair[1]):
            new_indices.append(new_ind)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None,
            ):

        self.vocab = vocab
        self.inverse_vocab = {value: key for key, value in vocab.items()}
        self.merges = merges
        self.merges_int = [(self.inverse_vocab[merge[0]], self.inverse_vocab[merge[1]]) for merge in merges]
        self.special_tokens = special_tokens or []
        self.special_tokens = sorted(self.special_tokens, reverse=True)
        self.merge_rules = {
            (self.inverse_vocab[a], self.inverse_vocab[b]): (rank, self.inverse_vocab[a + b])
            for rank, (a, b) in enumerate(merges)
        }
        

    def encode(self, string: str) -> list[int]:
        strings_special_tokens = []
        if len(self.special_tokens) != 0:
            p = re.compile('|'.join(map(re.escape, self.special_tokens)))
            strings = p.split(string)
            strings_special_tokens = p.findall(string)
        else:
            strings = [string]
 
        res = []
        for i, substring in enumerate(strings):
            tokenize_text = re.finditer(PATTERN, substring)
            for token in tokenize_text:
                token_bytes = [bytes([b]) for b in token.group().encode("utf-8")]
                indices = [self.inverse_vocab[byte] for byte in token_bytes]
                while len(indices) > 1:
                    candidates = [(self.merge_rules[(a, b)][0], (a, b)) for a, b in zip(indices, indices[1:]) if (a, b) in self.merge_rules]
                    if not candidates:
                        break
                    _, best_pair = min(candidates)
                    new_id = self.merge_rules[best_pair][1]
                    indices = merge(indices, best_pair, new_id)
                res.extend(indices)
            if i < len(strings_special_tokens):
                res.append(self.inverse_vocab[strings_special_tokens[i].encode("utf-8")])
        return res

    def encode_parallel(self, texts: Iterable[str], num_workers: int = 4, chunksize: int = 100,) -> list[int]:
        total = len(texts) if hasattr(texts, "__len__") else None
        token_ids = []
        with mp.Pool(processes=num_workers) as pool:
            results = pool.imap(self.encode, texts, chunksize=chunksize)
            for ids in tqdm(results, total=total, desc="BPE tokenization"):
                token_ids.extend(ids)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8", errors='replace')
        return string


def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int, 
        special_tokens: list[str],
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = {x: bytes([x]) for x in range(256)}
    merges = []
    for i, special_token in enumerate(special_tokens):
        vocab[256+i] = special_token.encode("utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        lines_list = f.readlines()
    raw_text = "".join(lines_list)

    if len(special_tokens) != 0:
        p = re.compile('|'.join(map(re.escape, special_tokens)))
        raw_texts = p.split(raw_text)
    else:
        raw_texts = [raw_text]

    pretokenize_tokens_freq = {}
    for raw_text in raw_texts:
        tokenize_text = re.finditer(PATTERN, raw_text)
        for match in tokenize_text:
            key = tuple(match.group().encode("utf-8"))
            if key not in pretokenize_tokens_freq:
                pretokenize_tokens_freq[key] = 1
            else:
                pretokenize_tokens_freq[key] += 1

    for i in range(vocab_size - 256 - len(special_tokens)):
        new_ind = len(vocab)

        byte_pairs_freq = {}
        for pretokenize_tokens, freq in pretokenize_tokens_freq.items():
            for byte1, byte2 in zip(pretokenize_tokens, pretokenize_tokens[1:]):
                if (byte1, byte2) not in byte_pairs_freq:
                    byte_pairs_freq[(byte1, byte2)] = freq
                else:
                    byte_pairs_freq[(byte1, byte2)] += freq

        most_freq_pair = ()
        max_freq = 0
        for byte_pair, freq in byte_pairs_freq.items():
            if freq > max_freq:
                max_freq = freq
                most_freq_pair = byte_pair
            elif (freq == max_freq) and vocab[byte_pair[0]] > vocab[most_freq_pair[0]]:
                max_freq = freq
                most_freq_pair = byte_pair
            elif (freq == max_freq) and (vocab[byte_pair[0]] == vocab[most_freq_pair[0]]) and (vocab[byte_pair[1]] > vocab[most_freq_pair[1]]):
                max_freq = freq
                most_freq_pair = byte_pair

        if len(byte_pairs_freq) == 0:
            break

        vocab[new_ind] = vocab[most_freq_pair[0]] + vocab[most_freq_pair[1]]
        merges.append((vocab[most_freq_pair[0]], vocab[most_freq_pair[1]]))

        new_pretokenize_tokens_freq = {}
        for pretokenize_token, freq in pretokenize_tokens_freq.items():
            new_pretokenize_tokens_freq[tuple(merge(pretokenize_token, most_freq_pair, new_ind))] = freq
        pretokenize_tokens_freq = new_pretokenize_tokens_freq

    return vocab, merges



def create_bpe_tokenizer(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    with open("vocab.json", "w") as file:
        file.write(str(vocab))
    with open("merges.txt", "w", newline="") as file:
        file.write(str(merges))

    bpe_tokenizer = BPETokenizer(vocab, merges, special_tokens)
    return bpe_tokenizer
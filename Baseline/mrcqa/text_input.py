import itertools
import numpy as np
from nltk import sent_tokenize, word_tokenize


def rich_tokenize(text, vocab, c_vocab, update):
    tokens = list(
        itertools.chain.from_iterable(
            (token.replace("''", '"').replace("``", '"')
             for token in word_tokenize(sent))
            for sent in sent_tokenize(text)))
    length = len(tokens)
    mapping = np.zeros((length, 2), dtype='int32')
    c_lengths = np.zeros(length, dtype='int32')
    start = 0
    for ind, token in enumerate(tokens):
        _start = text.find(token, start)
        t_l = len(token)
        if _start < 0 and token[0] == '"':
            t_l = 2
            _a = text.find("''"+token[1:], start)
            _b = text.find("``"+token[1:], start)
            if _a != -1 and _b != -1:
                _start = min(_a, _b)
            elif _a != -1:
                _start = _a
            else:
                _start = _b
        start = _start
        assert start >= 0
        mapping[ind, 0] = start
        mapping[ind, 1] = start + t_l
        c_lengths[ind] = t_l
        start = start + t_l

    if update:
        character_ids = [
            [c_vocab.setdefault(c, len(c_vocab)) for c in token]
            for token in tokens]
        token_ids = [
            vocab.setdefault(token, len(vocab)) for token in tokens]
    else:
        character_ids = [
            [c_vocab.get(c, 1) for c in token]
            for token in tokens]
        token_ids = [
            vocab.get(token, 1) for token in tokens]

    return token_ids, character_ids, length, c_lengths, mapping


def pad_to_size(token_ids, character_ids, t_length, c_length):
    padded_tokens = np.zeros((1, t_length), dtype='int32')
    padded_characters = np.zeros((1, t_length, c_length), dtype='int32')
    padded_tokens[0, :len(token_ids)] = token_ids
    for ind, _chars in enumerate(character_ids):
        padded_characters[0, ind, :len(_chars)] = _chars
    return padded_tokens, padded_characters


def text_as_batch(text, vocab, c_vocab):
    tokens, chars, length, c_lengths, mapping = \
        rich_tokenize(text, vocab, c_vocab, update=False)
    tokens, chars = pad_to_size(tokens, chars, length, max(5, c_lengths.max()))
    length = np.array([length])
    return tokens, chars, length, mapping

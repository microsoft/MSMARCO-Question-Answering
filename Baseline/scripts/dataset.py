"""
Module used to manage data:
    Basic loading and structuring for training and testing;
    Tokenization, with vocabulary creation;
    Injection of new tokens into existing vocabulary (with random or
    pre-trained embeddings)
    Conversion into Dataset for training, etc.
"""
import torch
from torch.autograd import Variable
import numpy as np

from mrcqa.text_input import rich_tokenize


def load_data(source, span_only, answered_only):
    """
    Load the data, and use it to create example triples.
    Input is a dictionary, output is a list of example triples, with added
    query id (query ids are not unique, all answers are included).

    :param: source: input dictionary, loaded from a json file.
    :param: span_only (bool): only keeps answers that are span in a passage.
    :param: answered_only (bool): only keeps answered questions.

    :return: a list of (qid, passage, query, (answer_start, answer_stop))
            tuples.
    :return: filtered_out: a set of qid that were filtered by span_only
             and answered_only.
    If there's no answer, answer_start = 0, answer_stop = 0.
    If the answer is not a span, answer_start = 0, answer_stop = len(passage)
    """

    query_ids = source['query_id']
    queries = source['query']
    passages = source['passages']
    answers = source.get('answers', {})

    flat = ((qid, passages[qid], queries[qid], answers.get(qid))
            for qid in query_ids)

    organized, filtered_out = _organize(flat, span_only, answered_only)
    return organized, filtered_out


def _organize(flat, span_only, answered_only):
    """
    Filter the queries and consolidate the answer as needed.
    """
    filtered_out = set()

    organized = []

    for qid, passages, query, answers in flat:
        if answers is None and not answered_only:
            filtered_out.add(qid)
            continue  # Skip non-answered queries

        matching = set()
        for ans in answers:
            if len(ans) == 0:
                continue
            for ind, passage in enumerate(passages):
                pos = passage['passage_text'].find(ans)
                if pos >= 0:
                    matching.add(ind)
                    organized.append((qid, passage, query,
                                      (pos, pos+len(ans))))
        # OK, found all spans.
        if not span_only or not answered_only:
            for ind, passage in enumerate(passages):
                if ind in matching:
                    continue
                if passage.get('is_selected', False):
                    matching.add(ind)
                    organized.append((qid, passage, query,
                                      (0, len(passage))))
                elif not answered_only:
                    matching.add(ind)
                    organized.append((qid, passage, query,
                                      (0, 0)))
        # Went through the whole thing. If there's still not match, then it got
        # filtered out.
        if len(matching) == 0:
            filtered_out.add(qid)

    if len(filtered_out) > 0:
        assert span_only or answered_only

    return organized, filtered_out


def tokenize_data(data, token_to_id, char_to_id, limit=None):
    """
    Tokenize a data set, with mapping of tokens to index in origin.
    Also create and update the vocabularies.

    :param: data: a flat, organize view of the data, as a list of qid, passage,
            query and answer indexes.
    :param: vocab: a dict of token to id; updated.
    :param: c_vocab: a dict of char to id; update.

    :return: a tokenized view of the data, as a list of qid, passage, query,
    answer indexes, and token to char indexes mapping.
    Passage and queries are tokenized into a tuple (token, chars).
    Answer indexes are start:stop range of tokens.
    """
    tokenized = []
    for qid, passage, query, (start, stop) in data:
        q_tokens, q_chars, _, _, _ = \
            rich_tokenize(query, token_to_id, char_to_id, update=True)
        p_tokens, p_chars, _, _, mapping = \
            rich_tokenize(passage['passage_text'],
                          token_to_id, char_to_id, update=True)

        if start == 0 and stop == 0:
            pass  # No answer; nop, since 0 == 0
        elif start == 0 and stop == len(passage):
            stop = len(p_tokens)  # Now point to just after last token.
        else:
            t_start = None
            t_end = len(p_tokens)
            for t_ind, (_start, _end) in enumerate(mapping):
                if start < _end:
                    t_start = t_ind
                    break
            assert t_start is not None
            for t_ind, (_start, _end) in \
                    enumerate(mapping[t_start:], t_start):
                if stop < _start:
                    t_end = t_ind
                    break
            start = t_start  # Now point to first token in answer.
            stop = t_end  # Now point to after the last token in answer.

        # Keep or not based on length of passage.
        if limit is not None and len(p_tokens) > limit:
            if stop <= limit:
                # Passage is too long, but it can be trimmed.
                p_tokens = p_tokens[:limit]
            else:
                # Passage is too long, but it cannot be trimmed.
                continue

        tokenized.append(
            (qid,
             (p_tokens, p_chars),
             (q_tokens, q_chars),
             (start, stop),
             mapping))

    return tokenized


def symbol_injection(id_to_symb, start_at, embedding,
                     pre_trained_source, random_source):
    """
    Inject new symbols into an embedding.
    If possible, the new embedding are retrieved from a pre-trained source.
    Otherwise, they get a new random value, using the random source.

    Will also overwrite embedding[start_at:].
    """
    if start_at == len(id_to_symb):
        return embedding  # Nothing to do.
    dim = embedding.shape[1]
    assert start_at <= len(id_to_symb)
    assert start_at <= len(embedding)
    if start_at > 0:
        embedding = embedding[:start_at]
        augment_by = len(id_to_symb) - start_at
        augment = np.empty((augment_by, dim), dtype=embedding.dtype)
        embedding = np.concatenate((embedding, augment), axis=0)
    else:
        embedding = np.empty((len(id_to_symb), dim), dtype=embedding.dtype)

    for id_ in range(start_at, len(id_to_symb)):
        symbol = id_to_symb[id_]
        embedding[id_] = pre_trained_source(symbol, dim, random_source)
    return embedding


class SymbolEmbSource(object):
    """
    Base class for symbol embedding source.
    """

    def __init__(self):
        return

    def get_rep(self, symbol, dim):
        return None

    def __call__(self, symbol, dim, fallback=None):
        rep = self.get_rep(symbol, dim)
        if rep is None and fallback is not None:
            rep = fallback(symbol, dim)

        if rep is None:
            raise ValueError('Symbol [%s] cannot be found' % str(symbol))
        return rep


class SymbolEmbSourceText(SymbolEmbSource):
    """
    Load pre-trained embedding from a file object, saving only symbols of
    interest.
    If none, save all symbols.
    The saved symbols can then be retrieves

    Assumes that base_file contains line, with one symbol per line.
    """

    def __init__(self, base_file, symbols_of_interest, dtype='float32'):
        self.symbols = {}

        if symbols_of_interest is not None:
            def _skip(symbol):
                return symbol not in symbols_of_interest
        else:
            def _skip(symbol):
                return False

        dim = None
        for line in base_file:
            line = line.strip().split()
            if not line or _skip(line[0]):
                continue
            if dim is None:
                dim = len(line) - 1
            else:
                if len(line) != dim + 1:
                    continue
            symbol = line[0]
            rep = np.array([float(v) for v in line[1:]], dtype='float32')
            self.symbols[symbol] = rep
        return

    def get_norm_stats(self, use_cov=True):
        """
        Assumes that the representation are normally distributed, and return
        the mean and standard deviation of that distribution.
        """
        data = np.array(list(self.symbols.values()))
        if use_cov:
            cov = np.cov(data, rowvar=False)
        else:
            cov = data.std(0)
        return data.mean(0), cov

    def get_rep(self, symbol, dim):
        """
        Get the representation for a symbol, and confirm that the dimensions
        match. If everything matches, return the representation, otherwise
        return None.
        """
        rep = self.symbols.get(symbol)
        if rep is not None and len(rep) != dim:
            rep = None
        return rep


class SymbolEmbSourceNorm(SymbolEmbSource):
    """
    Create random representation for symbols.
    """

    def __init__(self, mean, cov, rng, use_voc=False, cache=False):

        self.mean = mean
        self.cov = cov
        self.use_voc = use_voc
        self.rng = rng
        self.cache = {} if cache else None
        return

    def get_rep(self, symbol, dim):
        if dim != len(self.mean):
            return None
        if self.cache and symbol in self.cache:
            return self.cache[symbol]
        if self.use_voc:
            rep = self.rng.multivariate_normal(self.mean, self.cov)
        else:
            rep = self.rng.normal(self.mean, self.cov)

        if self.cache is not None:
            self.cache[symbol] = rep
        return rep


class EpochGen(object):
    """
    Generate batches over one epoch.
    """

    def __init__(self, data, batch_size=32, shuffle=True,
                 tensor_type=torch.LongTensor):
        """
        Parameters:
            :param: data: The dataset from which the data
            is taken, and whose size define the epoch length
            :param: batch_size (int): how many questions should be included in
            a batch. Default to 32
            :param: shuffle (bool): Should the data be shuffled at the start of
            the epoch. Default to True.
            :param: tensor_type (class): The type of the tensors. Default to
            LongTensor, i.e. Long integer on CPU.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tensor_type = tensor_type
        self.n_samples = len(data)
        self.idx = np.arange(self.n_samples)
        self.data = data
        return

    def process_batch_for_length(self, sequences, c_sequences):
        """
        Assemble and pad data.
        """
        assert len(sequences) == len(c_sequences)
        lengths = Variable(self.tensor_type([len(seq) for seq in sequences]))
        max_length = max(len(seq) for seq in sequences)
        max_c_length = max(max(len(chars) for chars in seq)
                           for seq in c_sequences)

        def _padded(seq, max_length):
            _padded_seq = self.tensor_type(max_length).zero_()
            _padded_seq[:len(seq)] = self.tensor_type(seq)
            return _padded_seq
        sequences = Variable(torch.stack(
                [_padded(seq, max_length) for seq in sequences]))

        def _padded_char(seq, max_length, max_c_length):
            _padded = self.tensor_type(max_length, max_c_length).zero_()
            for ind, tok in enumerate(seq):
                _padded[ind, :len(tok)] = self.tensor_type(tok)
            return _padded

        c_sequences = Variable(torch.stack([
            _padded_char(seq, max_length, max_c_length)
            for seq in c_sequences]))

        return (sequences, c_sequences, lengths)

    def __len__(self):
        return (self.n_samples + self.batch_size - 1)//self.batch_size

    def __iter__(self):
        """
        Generate batches from data.
        All outputs are in torch.autograd.Variable, for convenience.
        """

        if self.shuffle:
            np.random.shuffle(self.idx)

        for start_ind in range(0, self.n_samples - 1, self.batch_size):
            batch_idx = self.idx[start_ind:start_ind+self.batch_size]

            qids = [self.data[ind][0] for ind in batch_idx]
            passages = [self.data[ind][1][0] for ind in batch_idx]
            c_passages = [self.data[ind][1][1] for ind in batch_idx]
            queries = [self.data[ind][2][0] for ind in batch_idx]
            c_queries = [self.data[ind][2][1] for ind in batch_idx]
            answers = [self.data[ind][3] for ind in batch_idx]
            mappings = [self.data[ind][4] for ind in batch_idx]

            passages = self.process_batch_for_length(
                    passages, c_passages)
            queries = self.process_batch_for_length(
                    queries, c_queries)

            answers = Variable(self.tensor_type(answers))

            batch = (qids,
                     passages, queries,
                     answers,
                     mappings)
            yield batch
        return

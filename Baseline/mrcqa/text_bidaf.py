import torch
from torch.autograd import Variable
import numpy as np

import mrcqa
from mrcqa.bidaf import BidafModel
from mrcqa.text_input import rich_tokenize, pad_to_size


class TextBidaf(BidafModel):

    def __init__(self, embedder,
                 num_highways,
                 num_lstm,
                 hidden_size,
                 dropout, vocab, c_vocab):
        super(TextBidaf, self).__init__(
            embedder, num_highways, num_lstm,
            hidden_size, dropout)
        self.vocab = vocab
        self.c_vocab = c_vocab
        self.tensor_type = torch.LongTensor
        return

    def cuda(self, *args):
        self.tensor_type = torch.cuda.LongTensor
        super(TextBidaf, self).cuda(*args)
        return

    def cpu(self):
        self.tensor_type = torch.LongTensor
        super(TextBidaf, self).cpu()
        return

    def _to_batch(self, texts):
        mappings = []
        lengths = []
        c_lengths = []
        tokens = []
        chars = []
        for text in texts:
            _tokens, _chars, length, _c_lengths, mapping = \
                rich_tokenize(text,
                              self.vocab,
                              self.c_vocab, update=False)
            mappings.append(mapping)
            lengths.append(length)
            c_lengths.append(_c_lengths)
            tokens.append(_tokens)
            chars.append(_chars)

        lengths = np.array(lengths)
        p_length = lengths.max()
        p_c_length = max(max(_c_lengths) for _c_lengths in c_lengths)

        b_tokens = []
        b_chars = []
        for _tokens, _chars in zip(tokens, chars):
            _tokens, _chars = pad_to_size(_tokens, _chars,
                                          p_length,
                                          max(5, p_c_length))
            b_tokens.append(_tokens)
            b_chars.append(_chars)

        b_tokens = np.concatenate(b_tokens)
        b_chars = np.concatenate(b_chars)

        return b_tokens, b_chars, lengths, mappings

    def forward(self, passages, questions):
        """
        Tokenize passage and question, then find the best span answering the
        question.

        Parameters:
            :param: passages (list of str, unicode): the passages
            :param: questions (list of str, unicode): the questions

        Returns:
            :return: answers (list str, unicode): the text from passage
            corresponding to the answer
            :return: starts: the start index of the answer in passage
            :return: ends: the end index of the answer in passage, so that
            passage[i, starts[i]:ends[i]] == answers[i]
            :return: losses: the negative log-likelihood of the answer
        """
        p_t, p_c, p_l, mappings = self._to_batch(passages)
        q_t, q_c, q_l, _ = self._to_batch(questions)

        _passage = [
            Variable(self.tensor_type(p_t)),
            Variable(self.tensor_type(p_c))]
        p_lengths = Variable(self.tensor_type(p_l))
        _question = [
            Variable(self.tensor_type(q_t)),
            Variable(self.tensor_type(q_c))]
        q_lengths = Variable(self.tensor_type(q_l))

        starts, ends = super(TextBidaf, self).forward(
            _passage, p_lengths, _question, q_lengths)
        # Ok until here; need check below.

        bests = self.get_best_span(starts, ends)
        _bests = Variable(
            bests.type(self.tensor_type))
        losses = []
        for _start, _end, _best in zip(starts, ends, _bests):
            _start = _start.unsqueeze(0)
            _end = _end.unsqueeze(0)
            _best = _best.unsqueeze(0)
            losses.append(
                self.get_loss(
                    _start, _end,
                    _best[:, 0], _best[:, 1]).cpu().data[0])

        answers = []
        start_chars = []
        end_chars = []
        for mapping, best, passage in zip(mappings, bests, passages):
            start_char = mapping[best[0], 0]
            end_char = mapping[best[1]-1, 1]
            answer = passage[start_char:end_char]
            answers.append(answer)
            start_chars.append(start_char)
            end_chars.append(end_char)

        return answers, start_chars, end_chars, losses

    @classmethod
    def _parse_config(cls, config, vocab, c_vocab):
        args = super()._parse_config(config, vocab, c_vocab)
        word2id = {w: i for i, w in enumerate(vocab)}
        char2id = {c: i for i, c in enumerate(c_vocab)}
        args = args + (word2id, char2id)
        return args

"""
BiDAF model for MRC.
"""

import torch
from torch import nn
from torch.nn.functional import nll_loss
from torch.autograd import Variable
import numpy as np

import mrcqa
from mrcqa.modules.highway import Highways


class BidafModel(nn.Module):
    """
    Bidirectional attention flow model for question answering.
    """

    def __init__(self,
                 embedder,
                 num_highways,
                 num_lstm,
                 hidden_size,
                 dropout):
        """
        Create a BiDAF model. The input is a tensor of indices, or a tuple of
        same. The outputs are start and end log probability vectors..

        Overall model, assuming no batches:
            1. The passage and question are encoded independently using a
            shared set of embeddings, highway layers and a bidirectional
            LSTM layer.
            2. The passage and question are combined into an attention matrix.
            3. The attention matrix is applied to the question, to get a
            question-in-passage matrix, with one row per token in the passage.
            4. The same attention matrix is applied to the passage, to get a
            passage-in-question vector, which is then tiled to get one row per
            token in the passage.
            5. The resulting matrices are concatenated with the passage, and
            with their product with the passage.
            6. This is then passed through a stack of bidirectional LSTMs.
            7. The results is projected down to 1 dimension, to get the start
            logits.
            8. This is also used as attention, and combined with the LSTM stack
            inputs and outputs, and passed through a final LSTM.
            9. The output is again concatenated with step 5, and projected down
            to 1 dimension, to get the end logits.
            10. A log-softmax is then applied to the logits.

        Parameters:
            :param: embedder (Module): the module in that will embed the
            passage and question
            :param: num_highways (int): the number of highway layers to use
            :param: num_lstm (int): the number of LSTM layers to use
            :param: hidden_size (int): The size of the hidden layers;
            effectively doubled for bidirectional LSTMs
            :param: dropout (float,>=0 or None) Dropout probability

        Variables/sub-modules:
            embedder: the embeddings
            highways: the highway layers
            seq_encoder: the LSTM used after the highway layers to get the
            passage and question representations
            attention: the module used to get the attention matrix
            extractor: the stack of LSTM following attention
            end_encoder: the final LSTM, used to get the end logits
            start_projection: the projection to get the start logits
            end_projection: the projection to get the end logits

        Input:
            :param: passage: features sent to embedder for the passages
            :param: p_lengths: vector containing the passage lengths
            :param: question: features sent to embedder for the questions
            :param: q_lengths: vector containing the question lengths

        Return:
            :return: start_log_probs: (batch, passage_size) float tensor
            containing the log probabilities of the start points
            :return: end_log_probs: (batch, passage_size) float tensor
            containing the log probabilities of the end points
        """
        super(BidafModel, self).__init__()
        self.hidden_size = hidden_size
        self.bidir_hidden_size = 2*hidden_size
        self.embedder = embedder
        self.highways = Highways(embedder.output_dim, num_highways)
        self.seq_encoder = nn.LSTM(embedder.output_dim,
                                   hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=0,
                                   bidirectional=True)
        self.extractor = nn.LSTM(4*self.bidir_hidden_size,
                                 hidden_size,
                                 num_layers=num_lstm,
                                 batch_first=True,
                                 dropout=0,
                                 bidirectional=True)
        self.end_encoder = nn.LSTM(7*self.bidir_hidden_size,
                                   hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=dropout,
                                   bidirectional=True)
        self.attention = AttentionMatrix(self.bidir_hidden_size)

        # Second hidden_size is for extractor.
        self.start_projection = nn.Linear(
            4*self.bidir_hidden_size + self.bidir_hidden_size, 1)
        self.end_projection = nn.Linear(
            4*self.bidir_hidden_size + self.bidir_hidden_size, 1)

        if dropout and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda nop: nop
        return

    @classmethod
    def _pack_and_unpack_lstm(cls, input, lengths, seq_encoder):
        """
        LSTM, when using batches, should be called with a PackedSequence.
        Doing this will deal with the different lengths in the batch.
        PackedSequence must be created from batches where the sequences are
        stored with decreasing lengths.

        _pack_and_unpack_lstm handles this issue.
        It re-orders its input, pack it, sends it through the LSTM and finally
        restore the original order.

        This is not general purpose: in particular, it does not handle initial
        and final states.
        """
        s_lengths, indexes = lengths.sort(0, descending=True)
        s_input = input.index_select(0, indexes)

        i_range = torch.arange(lengths.size()[0]).type_as(lengths.data)
        i_range = Variable(i_range)
        _, reverses = indexes.sort(0, descending=False)
        reverses = i_range.index_select(0, reverses)

        packed = nn.utils.rnn.pack_padded_sequence(
            s_input, s_lengths.data.tolist(), batch_first=True)

        output, _ = seq_encoder(packed)
        # Unpack and apply reverse index.
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)
        output = output.index_select(0, reverses)

        return output

    @classmethod
    def _apply_attention_to_question(cls, similarity, enc_question, mask):
        """
        Apply attention to question, while masking for lengths
        """
        # similarity: [batch, m_p, m_q]
        # enc_question: [batch, m_q, hidden_size]
        # mask: [batch, m_q]
        batch, m_p, m_q = similarity.size()

        _sim = similarity.view(
            batch*m_p, m_q)

        tmp_mask = mask.unsqueeze(1).expand(
            batch, m_p, m_q).contiguous().float()
        tmp_mask = tmp_mask.view(batch*m_p, m_q)
        _sim = nn.functional.softmax(_sim*tmp_mask - (1-tmp_mask)*1e20, dim=1)
        _sim = _sim.view(batch, m_p, m_q)

        out = _sim.bmm(enc_question)
        return out

    @classmethod
    def _apply_attention_to_passage(cls, similarity, enc_passage,
                                    p_mask, q_mask):
        """
        Apply attention to passage, while masking for lengths.
        """
        # similarity: [batch, m_p, m_q]
        # enc_passage: [batch, m_p, hidden_size]
        # p_mask: [batch, m_p]
        # q_mask: [batch, m_q]
        batch, m_p, m_q = similarity.size()

        # Mask the similarity
        tmp_mask = q_mask.unsqueeze(1).expand(
            batch, m_p, m_q).contiguous().float()
        similarity = similarity * tmp_mask - (1-tmp_mask)*1e20
        # Pick the token in the question with the highest similarity with a
        # given token in the passage as the similarity between the entire
        # question and that passage token
        similarity = similarity.max(dim=2)[0]
        # Final similarity: [batch, m_p]

        tmp_mask = (1-p_mask)
        tmp_mask = 1e20*tmp_mask
        similarity = nn.functional.softmax(similarity*p_mask - tmp_mask, dim=1)
        out = similarity.unsqueeze(1).bmm(enc_passage).squeeze(1)
        return out

    def _encode(self, features, lengths):
        """
        Encode text with the embedder, highway layers and initial LSTM.
        """
        embedded = self.embedder(features)
        batch_size, num_tokens = embedded.size()[:2]
        embedded = self.highways(embedded.view(
            batch_size*num_tokens, -1))
        embedded = embedded.view(batch_size, num_tokens, -1)
        encoded = self.dropout(self._pack_and_unpack_lstm(
            embedded, lengths, self.seq_encoder))
        return encoded

    @classmethod
    def _create_mask_like(cls, lengths, like):
        """
        Create masks based on lengths. The mask is then converted to match the
        type of `like`, a Variable.
        """
        mask = torch.zeros(like.size()[:2])
        for ind, _length in enumerate(lengths.data):
            mask[ind, :_length] = 1
        mask = mask.type_as(like.data)
        mask = Variable(mask, requires_grad=False)
        return mask

    def _attention(self, enc_passage, enc_question, p_mask, q_mask):
        """
        Get and apply the attention matrix for the passage and question.
        """
        batch_size, p_num_tokens = enc_passage.size()[:2]
        # Similarity score (unnormalized) between passage and question.
        # Shape: [batch, p_num_tokens, q_num_tokens]
        similarity = self.attention(enc_passage, enc_question)

        # Shape: [batch, p_num_tokens, hidden_size]
        question_in_passage = self._apply_attention_to_question(
            similarity, enc_question, q_mask)

        # Shape: [batch, hidden_size]
        passage_in_question = self._apply_attention_to_passage(
            similarity, enc_passage, p_mask, q_mask)
        passage_in_question = passage_in_question.unsqueeze(1).expand(
            batch_size, p_num_tokens, self.bidir_hidden_size)
        return question_in_passage, passage_in_question

    def forward(self, passage, p_lengths, question, q_lengths):
        """
        Forward pass
        """

        # Encode the text
        enc_passage = self._encode(passage, p_lengths)
        enc_question = self._encode(question, q_lengths)

        # Get the sizes
        batch_size, p_num_tokens = enc_passage.size()[:2]
        q_batch_size, q_num_tokens = enc_question.size()[:2]
        assert batch_size == q_batch_size
        assert batch_size == p_lengths.size()[0]
        assert batch_size == q_lengths.size()[0]

        # Create the masks
        p_mask = self._create_mask_like(p_lengths, enc_passage)
        q_mask = self._create_mask_like(q_lengths, enc_question)

        # Get similarities and apply the attention mechanism
        (question_in_passage, passage_in_question) = \
            self._attention(enc_passage, enc_question, p_mask, q_mask)

        # Concatenate the passage and similarities, then use a LSTM stack to
        # extract features.
        # 4 [b, p_num_tokens, hidden_size]
        # -> [b, n, 4*hidden_size]
        merged_passage = torch.cat([
            enc_passage,
            question_in_passage,
            enc_passage * question_in_passage,
            enc_passage * passage_in_question],
            dim=2)
        extracted = self.dropout(self._pack_and_unpack_lstm(
            merged_passage, p_lengths, self.extractor))

        # Use the features to get the start point probability vectors.
        # Also use it to as attention over the features.
        start_input = self.dropout(
            torch.cat([merged_passage, extracted], dim=2))
        # [b, p_num_tokens, 4*h] -> [b, n, 1] -> [b, n]
        start_projection = self.start_projection(start_input).squeeze(2)
        # Mask
        start_logits = start_projection*p_mask + (p_mask-1)*1e20
        # And turns into probabilities
        start_probs = nn.functional.softmax(start_logits, dim=1)
        # And then into representation, as attention.
        # [b, 1, hidden_size] -> [b, p_num_tokens, hidden_size]
        start_reps = start_probs.unsqueeze(1).bmm(extracted)
        start_reps = start_reps.expand(
            batch_size, p_num_tokens, self.bidir_hidden_size)

        # Uses various level of features to create the end point probability
        # vectors.
        # [b, n, 7*hidden_size]
        end_reps = torch.cat([
            merged_passage,
            extracted,
            start_reps,
            extracted * start_reps],
            dim=2)
        enc_end = self.dropout(self._pack_and_unpack_lstm(
            end_reps, p_lengths, self.end_encoder))
        end_input = self.dropout(torch.cat([
            merged_passage, enc_end], dim=2))
        # [b, p_num_tokens, 7*h] -> [b, n, 1] -> [b, n]
        end_projection = self.end_projection(end_input).squeeze(2)
        # Mask
        end_logits = end_projection*p_mask + (p_mask-1)*1e20

        # Applies the final log-softmax to get the actual log-probability
        # vectors.
        start_log_probs = nn.functional.log_softmax(start_logits, dim=1)
        end_log_probs = nn.functional.log_softmax(end_logits, dim=1)

        return start_log_probs, end_log_probs

    @classmethod
    def get_loss(cls, start_log_probs, end_log_probs, starts, ends):
        """
        Get the loss, $-\log P(s|p,q)P(e|p,q)$.
        The start and end labels are expected to be in span format,
        so that text[start:end] is the answer.
        """

        # Subtracts 1 from the end points, to get the exact indices, not 1
        # after the end.
        loss = nll_loss(start_log_probs, starts) +\
            nll_loss(end_log_probs, ends-1)
        return loss

    @classmethod
    def get_best_span(cls, start_log_probs, end_log_probs):
        """
        Get the best span.
        """
        if isinstance(start_log_probs, Variable):
            start_log_probs = start_log_probs.data
        if isinstance(end_log_probs, Variable):
            end_log_probs = end_log_probs.data

        batch_size, num_tokens = start_log_probs.size()
        start_end = torch.zeros(batch_size, 2).long()
        max_val = start_log_probs[:, 0] + end_log_probs[:, 0]
        max_start = start_log_probs[:, 0]
        arg_max_start = torch.zeros(batch_size).long()

        for batch in range(batch_size):
            _start_lp = start_log_probs[batch]
            _end_lp = end_log_probs[batch]
            for t_s in range(1, num_tokens):
                if max_start[batch] < _start_lp[t_s]:
                    arg_max_start[batch] = t_s
                    max_start[batch] = _start_lp[t_s]

                cur_score = max_start[batch] + _end_lp[t_s]
                if max_val[batch] < cur_score:
                    start_end[batch, 0] = arg_max_start[batch]
                    start_end[batch, 1] = t_s
                    max_val[batch] = cur_score

        # Place the end point one time step after the end, so that
        # passage[s:e] works.
        start_end[:, 1] += 1
        return start_end

    @classmethod
    def get_combined_logits(cls, start_log_probs, end_log_probs):
        """
        Combines the start and end log probability vectors into a matrix.
        The rows correspond to start points, the columns to end points.
        So, the value at m[s,e] is the log probability of the span from s to e.
        """
        batch_size, p_num_tokens = start_log_probs.size()

        t_starts = start_log_probs.unsqueeze(2).expand(
            batch_size, p_num_tokens, p_num_tokens)
        t_ends = end_log_probs.unsqueeze(1).expand(
            batch_size, p_num_tokens, p_num_tokens)
        return t_starts + t_ends

    @classmethod
    def get_combined_loss(cls, combined, starts, ends):
        """
        Get the loss, $-\log P(s,e|p,q)$.
        In practice, with:
            1. $\Psi_s(s|p,q)$ the start logits,
            2. $\Psi_e(e|p,q)$ the end logits,
            3. $Z_s = \log\sum_{i}\exp\Psi_s(i|p,q)$, the start partition,
            4. $Z_e = \log\sum_{i}\exp\Psi_e(i|p,q)$, the end partition, and
            5. $Z_c = \log\sum_{i}\sum{j>=i}\exp(\Psi_s(i|p,q)+\Psi_e(i|p,q))$,
            the combined partition,
        the default loss is:
            $Z_s + Z_e - \Psi_s(s|p,q) - \Psi_e(e|p,q)$,
        and the combined loss is:
            $Z_c - \Psi_s(s|p,q) - \Psi_e(e|p,q)$.

        The combined loss uses a normalization that ignores invalid end points.
        This is not a major difference, and should only slow things down during
        training.
        This loss is only used to validate and to compare.
        """
        batch_size, num_tokens, _other = combined.size()
        assert num_tokens == _other
        mask = torch.zeros(batch_size, num_tokens, num_tokens).float()
        for start in range(1, num_tokens):
            mask[:, start, :start] = -1e20
        mask = mask.type_as(combined.data)
        combined = combined + Variable(mask)
        combined = combined.view(batch_size, num_tokens*num_tokens)
        combined = nn.functional.log_softmax(combined, dim=1)
        labels = starts * num_tokens + ends
        return nll_loss(combined, labels)

    @classmethod
    def _parse_config(cls, config, vocab, c_vocab):
        num_tokens = len(vocab)
        num_chars = len(c_vocab)

        token_embs = mrcqa.modules.TokenEmbedding(
            num_tokens, config['embedding_dim'],
            output_dim=config.get('embedding_reduce'))

        _config = config['characters']
        char_embs = mrcqa.modules.CharEmbedding(
            num_chars,
            _config.get('dim', 16),
            _config.get('num_filters', 100),
            _config.get('filter_sizes', [5]))
        args = (
                mrcqa.modules.CatEmbedding([token_embs, char_embs]),
                config.get('num_highways', 2),
                config.get('num_lstm', 2),
                config.get('hidden_size', 100),
                config.get('dropout', 0.2))
        return args

    @classmethod
    def from_config(cls, config, vocab, c_vocab):
        """
        Create a model using the model description in the configuration file.
        """
        model = cls(*cls._parse_config(config, vocab, c_vocab))
        return model

    @classmethod
    def from_checkpoint(cls, config, checkpoint):
        """
        Load a model, on CPU and eval mode.

        Parameters:
            :param: config: a dictionary with the model's configuration
            :param: checkpoint: a h5 files containing the model's parameters.

        Returns:
            :return: the model, on the cpu and in evaluation mode.

        Example:
            ```
            with open('config.yaml') as f_o:
                config = yaml.load(f_o)

            with closing(h5py.File('checkpoint.h5', mode='r')) as checkpoint:
                model, vocab, c_vocab = BidafModel.from_checkpoint(
                    config, checkpoint)
            model.cuda()
            ```
        """
        model_vocab = checkpoint['vocab']
        model_c_vocab = checkpoint['c_vocab']

        model_vocab = {id_: tok for id_, tok in enumerate(model_vocab)}
        model_c_vocab = {id_: tok for id_, tok in enumerate(model_c_vocab)}

        model = cls.from_config(
                config,
                model_vocab,
                model_c_vocab)

        model.load_state_dict({
            name: torch.from_numpy(np.array(val))
            for name, val in
            checkpoint['model'].items()})
        model.cpu().eval()
        return model, model_vocab, model_c_vocab


class AttentionMatrix(nn.Module):
    """
    Attention Matrix (unnormalized)
    """

    def __init__(self, hidden_size):
        """
        Create a module for attention matrices. The input is a pair of
        matrices, the output is a matrix containing similarity scores between
        pairs of element in the matrices.

        Similarity between two vectors `a` and `b` is measured by
        $f(a, b) = W[a;b;ab] + C$, where:
            1. $W$ is a 1-by-3H matrix,
            2. $C$ is a bias,
            3. $ab$ is the element-wise product of $a$ and $b$.


        Parameters:
            :param: hidden_size (int): The size of the vectors

        Variables/sub-modules:
            projection: The linear projection $W$, $C$.

        Inputs:
            :param: mat_0 ([batch, n, hidden_size] Tensor): the first matrices
            :param: mat_1 ([batch, m, hidden_size] Tensor): the second matrices

        Returns:
            :return: similarity (batch, n, m) Tensor: the similarity matrices,
            so that similarity[:, n, m] = f(mat_0[:, n], mat_1[:, m])
        """
        super(AttentionMatrix, self).__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Linear(3*hidden_size, 1)
        return

    def forward(self, mat_0, mat_1):
        """
        Forward pass.
        """
        batch, n_0, _ = mat_0.size()
        _, n_1, _ = mat_1.size()
        mat_0, mat_1 = self.tile_to_match(mat_0, mat_1)
        mat_p = mat_0*mat_1
        combined = torch.cat((mat_0, mat_1, mat_p), dim=3)
        # projected down to [b, n, m]
        projected = self.projection(
            combined.view(batch*n_0*n_1, 3*self.hidden_size))
        projected = projected.view(batch, n_0, n_1)
        return projected

    @classmethod
    def tile_to_match(cls, mat_0, mat_1):
        """
        Enables broadcasting between mat_0 and mat_1.
        Both are tiled to 4 dimensions, from 3.

        Shape:
            mat_0: [b, n, e], and
            mat_1: [b, m, e].

        Then, they get reshaped and expanded:
            mat_0: [b, n, e] -> [b, n, 1, e] -> [b, n, m, e]
            mat_1: [b, m, e] -> [b, 1, m, e] -> [b, n, m, e]
        """
        batch, n_0, size = mat_0.size()
        batch_1, n_1, size_1 = mat_1.size()
        assert batch == batch_1
        assert size_1 == size
        mat_0 = mat_0.unsqueeze(2).expand(
            batch, n_0, n_1, size)
        mat_1 = mat_1.unsqueeze(1).expand(
            batch, n_0, n_1, size)
        return mat_0, mat_1

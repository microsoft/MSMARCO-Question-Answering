"""
Embeddings for characters and tokens, as well as concatenation of embeddings.
"""

import torch
from torch import nn


class CharEmbedding(nn.Module):
    """
    Convolutional character embedding.
    """

    PAD_CHAR = 0

    def __init__(self, num_embeddings, embedding_dim,
                 num_filters,
                 filter_sizes=(2, 3, 4, 5),
                 activation_class=nn.ReLU,
                 output_dim=None):
        """
        Create an embedding for the characters found in a word. The input is a
        tensor of character indices, the output is a tensor of embeddings.

        Parameters:
            :param: num_embeddings (int): the size of the dictionary (i.e. the
            number of characters)
            :param: embedding_dim (int): the size of the embedding vectors
            :param: num_filters (int): the number of filter to use
            :param: filter_size (int tuple): kernel size of the filters,
            equivalent to the char-n-grams used. Default: (2, 3, 4, 5)
            :param: activation_class (nn.Module): the class of the activation
            function for the convolutional layers. Default: ReLU
            :param: output_dim (int): the size of the final token
            representation. If specified, a linear projection is added.
            If not specified, num_filters*len(filter_sizes), without an extra
            projection.


        Variables/sub-modules:
            embeddings: the base char embedding, used before the max pooling
            layers: list of convolutional layers

        Input:
            :param: characters: tensor of char index with shape [batch,
            num_tokens, num_chars]

        Return:
            :return: tensor of embedded tokens with shape [batch, num_tokens,
            self.output_dim]
        """
        super(CharEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.activation = activation_class()

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

        # convolution step
        self.layers = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim,
                       out_channels=num_filters,
                       kernel_size=_size)
             for _size in filter_sizes])

        base_output_dim = num_filters*len(self.layers)
        if output_dim:
            self.output_dim = output_dim
            self.projection = nn.Linear(base_output_dim, output_dim)
        else:
            self.output_dim = base_output_dim
            self.projection = None

        return

    def forward(self, characters):
        """
        Encode the tokens based on the characters.
        """
        batch, num_tokens, num_chars = characters.size()
        char_emb = self.embeddings(
            characters.view(batch*num_tokens, num_chars))
        # char_emb: [b*t, c, emb]

        # Mask for padding.
        mask = (characters != self.PAD_CHAR)
        mask = mask.view(batch*num_tokens, num_chars, 1).float()
        char_emb = char_emb * mask

        # Transpose to match Conv1d, which expects [b*t, emb, c]
        char_emb = torch.transpose(char_emb, 1, 2)

        # Apply the convolution, and perform max pooling.
        # _outputs: [[b*t, num_filter]]
        _outputs = [
            self.activation(_layer(char_emb)).max(dim=2)[0]
            for _layer in self.layers]

        if len(_outputs) > 1:
            output = torch.cat(_outputs, dim=1)
        else:
            output = _outputs[0]
        # output: [batch*t, num_filter*len(filter_size)]

        if self.projection:
            output = self.projection(output)
        # output: [batch*t, output_dim]

        # Finally, unravel the time dimension.
        output = output.view(batch, num_tokens, self.output_dim)

        return output


class TokenEmbedding(nn.Module):
    """
    Token embedding.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 output_dim=None, static=True):
        """
        Create an embedding for the tokens in a sequence.
        The input is a tensor of token indices, the output is the embedding of
        each tokens.

        Parameters:
            :param: num_embeddings (int): the size of the dictionary (i.e. the
            number of tokens)
            :param: embedding_dim (int): the size of the embedding vectors
            :param: output_dim (int): the size of the final token
            representation. If specified, a linear projection is added.
            If not specified, use embedding_dim, without an extra
            projection.
            :param: static (bool): if True, the embeddings are not updated.
            Default to True. Note that any projection would still be updated.


        Variables/sub-modules:
            embeddings: the base token embedding

        Input:
            :param: characters: tensor of token indices with shape [batch,
            num_tokens]

        Return:
            :return: tensor of embedded tokens with shape [batch, num_tokens,
            self.output_dim]
        """
        super(TokenEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        if static:
            for param in self.embeddings.parameters():
                param.requires_grad = False

        if output_dim:
            self.output_dim = output_dim
            self.projection = nn.Linear(embedding_dim, output_dim)
        else:
            self.output_dim = embedding_dim
            self.projection = None

        return

    def forward(self, tokens):
        """
        Get the embeddings for tokens
        """
        batch, num_tokens = tokens.size()
        token_emb = self.embeddings(tokens)
        # token_emb: [b, t, emb]
        if self.projection:
            token_emb = self.projection(token_emb.view(
                batch*num_tokens, self.embedding_dim))
            token_emb = token_emb.view(batch, num_tokens, self.output_dim)
        # output: [batch, t, output_dim]

        return token_emb


class StaticTokenEmbedding(TokenEmbedding):
    """
    Convenience class to create static token embeddings.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(StaticTokenEmbedding, self).__init__(
            num_embeddings, embedding_dim, None, True)
        return


class UpdatedTokenEmbedding(TokenEmbedding):
    """
    Convenience class to create updated token embeddings.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(UpdatedTokenEmbedding, self).__init__(
            num_embeddings, embedding_dim, None, False)
        return


class CatEmbedding(nn.Module):
    """
    Concatenate embeddings together, possibly with a final projection.
    """

    def __init__(self, embeddings, output_dim=None):
        """
        Create an embedding for the tokens in a sequence by concatenating other
        embeddings.
        The input is a tuple of tensor, matching the corresponding embeddings.
        The output is the embedding of each tokens.

        Parameters:
            :param: embeddings (tuple of Embedding): the base embeddings.
            :param: output_dim (int): the size of the final token
            representation. If specified, a linear projection is added.
            If not specified, use sum(emb.output_dim for emb in
            embeddings), without projection.


        Variables/sub-modules:
            embeddings: list of base embeddings

        Input:
            :param: features: a tuple of tensors with one entry per base
            embedding, whose first dimensions are [batch, num_tokens]

        Return:
            :return: tensor of embedded tokens with shape [batch, num_tokens,
            self.output_dim]
        """
        super(CatEmbedding, self).__init__()
        self.embeddings = nn.ModuleList(embeddings)

        embedding_dim = sum(_emb.output_dim for _emb in self.embeddings)
        if output_dim:
            self.output_dim = output_dim
            self.projection = nn.Linear(embedding_dim, output_dim)
        else:
            self.output_dim = embedding_dim
            self.projection = None
        return

    def forward(self, features):
        if len(features) != len(self.embeddings):
            raise ValueError('CatEmbedding: mismatch between number of'
                             ' features and number of embedding',
                             len(features), len(self.embeddings))

        combined = torch.cat([
            _e(_f)
            for _f, _e in
            zip(features, self.embeddings)],
            dim=2)

        if self.projection:
            combined = self.projection(combined)

        return combined

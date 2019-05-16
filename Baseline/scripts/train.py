#!python3
"""
Training script: load a config file, create a new model using it,
then train that model.
"""
import json
import yaml
import argparse
import os.path
import itertools

import numpy as np
import torch
import torch.optim as optim
import h5py

from mrcqa import BidafModel

import checkpointing
from dataset import load_data, tokenize_data, EpochGen
from dataset import SymbolEmbSourceNorm
from dataset import SymbolEmbSourceText
from dataset import symbol_injection


def try_to_resume(force_restart, exp_folder):
    if force_restart:
        return None, None, 0
    elif os.path.isfile(exp_folder + '/checkpoint'):
        checkpoint = h5py.File(exp_folder + '/checkpoint')
        epoch = checkpoint['training/epoch'][()] + 1
        # Try to load training state.
        try:
            training_state = torch.load(exp_folder + '/checkpoint.opt')
        except FileNotFoundError:
            training_state = None
    else:
        return None, None, 0

    return checkpoint, training_state, epoch


def reload_state(checkpoint, training_state, config, args):
    """
    Reload state when resuming training.
    """
    model, id_to_token, id_to_char = BidafModel.from_checkpoint(
        config['bidaf'], checkpoint)
    if torch.cuda.is_available() and args.cuda:
        model.cuda()
    model.train()

    optimizer = get_optimizer(model, config, training_state)

    token_to_id = {tok: id_ for id_, tok in id_to_token.items()}
    char_to_id = {char: id_ for id_, char in id_to_char.items()}

    len_tok_voc = len(token_to_id)
    len_char_voc = len(char_to_id)

    with open(args.data) as f_o:
        data, _ = load_data(json.load(f_o),
                            span_only=True, answered_only=True)
    limit_passage = config.get('training', {}).get('limit')
    data = tokenize_data(data, token_to_id, char_to_id, limit_passage)

    data = get_loader(data, config)

    assert len(token_to_id) == len_tok_voc
    assert len(char_to_id) == len_char_voc

    return model, id_to_token, id_to_char, optimizer, data


def get_optimizer(model, config, state):
    """
    Get the optimizer
    """
    parameters = filter(lambda p: p.requires_grad,
                        model.parameters())
    optimizer = optim.Adam(
        parameters,
        lr=config['training'].get('lr', 0.01),
        betas=config['training'].get('betas', (0.9, 0.999)),
        eps=config['training'].get('eps', 1e-8),
        weight_decay=config['training'].get('weight_decay', 0))

    if state is not None:
        optimizer.load_state_dict(state)

    return optimizer


def get_loader(data, config):
    data = EpochGen(
        data,
        batch_size=config.get('training', {}).get('batch_size', 32),
        shuffle=True)
    return data


def init_state(config, args):
    token_to_id = {'': 0}
    char_to_id = {'': 0}
    print('Loading data...')
    with open(args.data) as f_o:
        data, _ = load_data(json.load(f_o), span_only=True, answered_only=True)
    print('Tokenizing data...')
    data = tokenize_data(data, token_to_id, char_to_id)
    data = get_loader(data, config)

    id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
    id_to_char = {id_: char for char, id_ in char_to_id.items()}

    print('Creating model...')
    model = BidafModel.from_config(config['bidaf'], id_to_token, id_to_char)

    if args.word_rep:
        print('Loading pre-trained embeddings...')
        with open(args.word_rep) as f_o:
            pre_trained = SymbolEmbSourceText(
                    f_o,
                    set(tok for id_, tok in id_to_token.items() if id_ != 0))
        mean, cov = pre_trained.get_norm_stats(args.use_covariance)
        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args.use_covariance)

        model.embedder.embeddings[0].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_token, 0,
                model.embedder.embeddings[0].embeddings.weight.data.numpy(),
                pre_trained, oovs))
    else:
        pass  # No pretraining, just keep the random values.

    # Char embeddings are already random, so we don't need to update them.

    if torch.cuda.is_available() and args.cuda:
        model.cuda()
    model.train()

    optimizer = get_optimizer(model, config, state=None)
    return model, id_to_token, id_to_char, optimizer, data


def train(epoch, model, optimizer, data, args):
    """
    Train for one epoch.
    """

    for batch_id, (qids, passages, queries, answers, _) in enumerate(data):
        start_log_probs, end_log_probs = model(
            passages[:2], passages[2],
            queries[:2], queries[2])
        loss = model.get_loss(
            start_log_probs, end_log_probs,
            answers[:, 0], answers[:, 1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


def main():
    """
    Main training program.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_folder", help="Experiment folder")
    argparser.add_argument("data", help="Training data")
    argparser.add_argument("--force_restart",
                           action="store_true",
                           default=False,
                           help="Force restart of experiment: "
                           "will ignore checkpoints")
    argparser.add_argument("--word_rep",
                           help="Text file containing pre-trained "
                           "word representations.")
    argparser.add_argument("--cuda",
                           type=bool, default=torch.cuda.is_available(),
                           help="Use GPU if possible")
    argparser.add_argument("--use_covariance",
                           action="store_true",
                           default=False,
                           help="Do not assume diagonal covariance matrix "
                           "when generating random word representations.")

    args = argparser.parse_args()

    config_filepath = os.path.join(args.exp_folder, 'config.yaml')
    with open(config_filepath) as f:
        config = yaml.load(f)

    checkpoint, training_state, epoch = try_to_resume(
            args.force_restart, args.exp_folder)

    if checkpoint:
        print('Resuming training...')
        model, id_to_token, id_to_char, optimizer, data = reload_state(
            checkpoint, training_state, config, args)
    else:
        print('Preparing to train...')
        model, id_to_token, id_to_char, optimizer, data = init_state(
            config, args)
        checkpoint = h5py.File(os.path.join(args.exp_folder, 'checkpoint'))
        checkpointing.save_vocab(checkpoint, 'vocab', id_to_token)
        checkpointing.save_vocab(checkpoint, 'c_vocab', id_to_char)

    if torch.cuda.is_available() and args.cuda:
        data.tensor_type = torch.cuda.LongTensor

    train_for_epochs = config.get('training', {}).get('epochs')
    if train_for_epochs is not None:
        epochs = range(epoch, train_for_epochs)
    else:
        epochs = itertools.count(epoch)

    for epoch in epochs:
        print('Starting epoch', epoch)
        train(epoch, model, optimizer, data, args)
        checkpointing.checkpoint(model, epoch, optimizer,
                                 checkpoint, args.exp_folder)

    return


if __name__ == '__main__':
    main()

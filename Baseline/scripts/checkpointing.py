"""
Modules for checkpointing functions.
"""
import h5py
import os.path

import torch
import numpy as np


def save_vocab(checkpoint, path, id_to_symb):
    """
    Save a vocab, that is id to symbol mapping, into a hp5y file.
    Used for checkpoints and such.
    """
    vocab = np.array([
        id_to_symb[ind] for ind in range(len(id_to_symb))])
    vocab = vocab.astype(h5py.special_dtype(vlen=str))
    if path in checkpoint:
        del checkpoint[path]

    checkpoint.create_dataset(path, data=vocab, compression='gzip')
    return


def save_params(destination, path, params):
    """
    Save a parameter tensor.
    """
    if path not in destination:
        destination.create_dataset(path, data=params, compression='gzip')
    else:
        destination[path][...] = params
    return


def save_model(model, destination):
    """
    Save a model into destination.
    """
    if 'model' not in destination:
        destination.create_group('model')

    for name, value in model.cpu().state_dict().items():
        save_params(destination, 'model/'+name, value)
    return


def save_epoch(epoch, destination):
    if 'training' not in destination:
        destination.create_dataset('training/epoch', data=epoch)
    else:
        destination['training/epoch'][()] = epoch
    return


def checkpoint(model, epoch, optimizer, dest, exp_folder):
    """
    Checkpoint the current state of training.
    """
    save_model(model, dest)
    save_epoch(epoch, dest)
    torch.save(optimizer.state_dict(),
               os.path.join(exp_folder, 'checkpoint.opt'))
    return

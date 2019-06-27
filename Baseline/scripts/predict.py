#!python3
"""
Training script: load a config file, create a new model using it,
then train that model.
"""
import json
import random
import yaml
import argparse
import os.path
import re
import numpy as np
import torch
import h5py
regex_drop_char                = re.compile('[^a-z0-9\s]+')
regex_multi_space              = re.compile('\s+')
from mrcqa import BidafModel

from dataset import load_data, tokenize_data, EpochGen
from dataset import SymbolEmbSourceNorm
from dataset import SymbolEmbSourceText
from dataset import symbol_injection


def try_to_resume(exp_folder):
    if os.path.isfile(exp_folder + '/checkpoint'):
        checkpoint = h5py.File(exp_folder + '/checkpoint')
    else:
        checkpoint = None
    return checkpoint


def reload_state(checkpoint, config, args):
    """
    Reload state before predicting.
    """
    print('Loading Model...')
    model, id_to_token, id_to_char = BidafModel.from_checkpoint(
        config['bidaf'], checkpoint)

    token_to_id = {tok: id_ for id_, tok in id_to_token.items()}
    char_to_id = {char: id_ for id_, char in id_to_char.items()}

    len_tok_voc = len(token_to_id)
    len_char_voc = len(char_to_id)

    with open(args.data) as f_o:
        data, _ = load_data(json.load(f_o), span_only=True, answered_only=True)
    data = tokenize_data(data, token_to_id, char_to_id)

    id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
    id_to_char = {id_: char for char, id_ in char_to_id.items()}

    data = get_loader(data, args)

    if len_tok_voc != len(token_to_id):
        need = set(tok for id_, tok in id_to_token.items()
                   if id_ >= len_tok_voc)

        if args.word_rep:
            with open(args.word_rep) as f_o:
                pre_trained = SymbolEmbSourceText(
                    f_o, need)
        else:
            pre_trained = SymbolEmbSourceText([], need)

        cur = model.embedder.embeddings[0].embeddings.weight.data.numpy()
        mean = cur.mean(0)
        if args.use_covariance:
            cov = np.cov(cur, rowvar=False)
        else:
            cov = cur.std(0)

        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args.use_covariance)

        if args.word_rep:
            print('Augmenting with pre-trained embeddings...')
        else:
            print('Augmenting with random embeddings...')

        model.embedder.embeddings[0].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_token, len_tok_voc,
                model.embedder.embeddings[0].embeddings.weight.data.numpy(),
                pre_trained, oovs))

    if len_char_voc != len(char_to_id):
        print('Augmenting with random char embeddings...')
        pre_trained = SymbolEmbSourceText([], None)
        cur = model.embedder.embeddings[1].embeddings.weight.data.numpy()
        mean = cur.mean(0)
        if args.use_covariance:
            cov = np.cov(cur, rowvar=False)
        else:
            cov = cur.std(0)

        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args.use_covariance)

        model.embedder.embeddings[1].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_char, len_char_voc,
                model.embedder.embeddings[1].embeddings.weight.data.numpy(),
                pre_trained, oovs))

    if torch.cuda.is_available() and args.cuda:
        model.cuda()
    model.eval()

    return model, id_to_token, id_to_char, data


def get_loader(data, args):
    data = EpochGen(
        data,
        batch_size=args.batch_size,
        shuffle=False)
    return data


def predict(model, data):
    """
    Train for one epoch.
    """
    for batch_id, (qids, passages, queries, _, mappings) in enumerate(data):
        start_log_probs, end_log_probs = model(
            passages[:2], passages[2],
            queries[:2], queries[2])
        predictions = model.get_best_span(start_log_probs, end_log_probs)
        predictions = predictions.cpu()
        passages = passages[0].cpu().data
        for qid, mapping, tokens, pred in zip(
                qids, mappings, passages, predictions):
            yield (qid, tokens[pred[0]:pred[1]],
                   mapping[pred[0], 0],
                   mapping[pred[1]-1, 1])
    return


def main():
    """
    Main prediction program.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_folder", help="Experiment folder")
    argparser.add_argument("data", help="Prediction data")
    argparser.add_argument("dest", help="Write predictions in")
    argparser.add_argument("--word_rep",
                           help="Text file containing pre-trained "
                           "word representations.")
    argparser.add_argument("--batch_size",
                           type=int, default=64,
                           help="Batch size to use")
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

    checkpoint = try_to_resume(args.exp_folder)

    if checkpoint:
        model, id_to_token, id_to_char, data = reload_state(
            checkpoint, config, args)
    else:
        print('Need a valid checkpoint to predict.')
        return

    if torch.cuda.is_available() and args.cuda:
        data.tensor_type = torch.cuda.LongTensor
    qid2candidate = {}
    for qid, toks, start, end in predict(model, data):
        toks = regex_multi_space.sub(' ', regex_drop_char.sub(' ', ' '.join(id_to_token[int(tok)] for tok in toks).lower())).strip()
        #print(repr(qid), repr(toks), start, end, file=f_o)
        output = '{\"query_id\": '+ qid + ',\"answers\":[ \"' + toks + '\"]}'
        if qid not in qid2candidate:
            qid2candidate[qid] = []
        qid2candidate[qid].append(json.dumps(json.loads(output)))
    with open(args.dest, 'w') as f_o:
        for qid in qid2candidate:
            #For our leaderboard model we build another model that predicted which passage would be most likley to produce the output. for simplicity we just pick one at random.
            pick = random.randint(0,len(qid2candidate[qid])-1)
            f_o.write(qid2candidate[qid][pick])
            f_o.write('\n')
    return


if __name__ == '__main__':
    main()

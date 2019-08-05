#!/bin/bash
# File Name : run.sh
#
# Description : This script downloads dependent scripts (if neccessary), 
# and compute evaluation metrics for MS MaRCo data set
#
# Command line: /ms_marco_metrics$ ./run.sh <path to reference file> <path to candidate file>
# Output: dictionary of {'bleu_n': <bleu_n score>, 'rouge_l': <rouge_l score>}
#
# Creation Date : Dec-15-2016
# Last Modified : Fri 16 December 2016 07:00:00 PT
# Authors : Tri Nguyen <trnguye@microsoft.com>, Xia Song <xiaso@microsoft.com>, Tong Wang <tongw@microsoft.com>

mkdir -p bleu
mkdir -p rouge

if [ ! -f "bleu/LICENSE" ]
then
    wget -O bleu/LICENSE https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/bleu/LICENSE
fi

if [ ! -f "bleu/bleu.py" ]
then
    wget -O bleu/bleu.py https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/bleu/bleu.py
    2to3 -wn bleu/bleu.py
fi

if [ ! -f "bleu/__init__.py" ]
then
    wget -O bleu/__init__.py https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/bleu/__init__.py
    2to3 -wn bleu/__init__.py
fi

if [ ! -f "bleu/bleu_scorer.py" ]
then
    wget -O bleu/bleu_scorer.py https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/bleu/bleu_scorer.py
    2to3 -wn bleu/bleu_scorer.py
fi

if [ ! -f "rouge/__init__.py" ]
then
    wget -O rouge/__init__.py https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/rouge/__init__.py
    2to3 -wn rouge/__init__.py
fi

if [ ! -f "rouge/rouge.py" ]
then
    wget -O rouge/rouge.py https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/rouge/rouge.py
    2to3 -wn rouge/rouge.py
fi

if [ ! $# -eq 2 ]
then
    echo "Invalid arguments supplied."
else
    PYTHONPATH=./bleu python3 ms_marco_eval.py $1 $2
fi

Microsoft MS MaRCo Evaluation
===================

Evaluation codes for MS MaRCo (Microsoft MAchine Reading COmprehension Dataset).

## Requirements ##
- python 3.5 : https://www.python.org/downloads/
- spacy: https://spacy.io/docs/usage/

## Instructions ##
Execute run.sh from /ms_marco_metrics/ in command line:
/ms_marco_metrics$ ./run.sh <path to reference json file> <path to candidate json file>
Example:
/ms_marco_metrics$ ./run.sh /home/trnguye/ms_marco_metrics/sample_test_data/sample_references.json /home/trnguye/ms_marco_metrics/sample_test_data/sample_candidates.json

Each line in both reference and candidate json files should be in format:
{"query_id": <a_query_id_int>, "answers": [<list_of_answers_string>]}
Note: <list_of_answers_string> must contain up to 1 answer in the candidate file.
Example (./sample_test_data/sample_references.json file):
{"query_id": 14509, "answers": ["It is include anemia, bleeding disorders such as hemophilia, blood clots, and blood cancers such as leukemia, lymphoma, and myeloma.", "HIV, hepatitis B, hepatitis C, and viral hemorrhagic fevers."]}
{"query_id": 14043, "answers": ["sp2", "sp2 hybridization"]}

Output from run.sh will be in the similar format to bellow:
bleu_1: 8.520511E-03
bleu_2: 4.666876E-10
bleu_3: 1.772338E-09
bleu_4: 3.453875E-09
rouge_l: 3.093306E-02

## Files ##
./
- ms_marco_eval.py: MS MaRCo Evaluation script.
- ms_marco_eval_test.py: Unit tests of ms_marco_eval.py .
- LICENSE
- run.sh: This script downloads dependent scripts, and compute evaluation metrics for MS MaRCo data set.

./sample_test_data
- dev_as_references.json : unit test input from dev set.
- dev_first_sentence_as_candidates.json : unit test with first sentence of first passage from dev set.
- no_answer_test_candidates.json : unit test input for no answer case.
- no_answer_test_references.json : unit test input for no answer case.
- same_answer_test_candidates.json : unit test input for same answer case.
- same_answer_test_references.json : unit test input for same answer case.
- sample_candidates.json : unit test input for sample data.
- sample_references.json : unit test input for sample data.

## References ##
- [Microsoft MAchine Reading COmprehension Dataset](https://arxiv.org/pdf/1611.09268v1.pdf).
- spaCy: We use [spaCy](https://spacy.io) for string tokenization and normalization.
- BLEU: We use [bleu-n calculation](https://github.com/tylin/coco-caption/tree/master/pycocoevalcap/bleu) from MS-COCO-caption; [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf).
- Rouge-L: We use [rouge-l calculation](https://github.com/tylin/coco-caption/tree/master/pycocoevalcap/rouge) from MS-COCO-caption; [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf).

## Developers ##
- Tri Nguyen <trnguye@microsoft.com>, Tong Wang <tongw@microsoft.com>, Xia Song <xiaso@microsoft.com>
import pandas as pd
import json
import sys
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: makeintermidiate.py <ground truth intermidiate file> <candidate intermidiate file> <output_filename>')
        exit()
    infile = sys.argv[1]
    infile2 = sys.argv[2]
    outfile = sys.argv[3]
    eval = pd.read_json(infile)
    ids = {}
    #identify all queries with well formed answers
    for row in eval.iterrows():
        if row[1]['wellFormedAnswers'] != '[]':
        	ids[row[1]['query_id']] = 1
    #rename columns
    eval = eval.drop('answers',1)
    eval = eval.rename(columns={'wellFormedAnswers':'answers'})
    eval = eval[eval.answers != '[]']
    #write modified well formed answer to format consumable by eval scripts
    with open('reference' + outfile,'w') as w:
        for row in eval.iterrows():
            w.write(str(row[1].to_json())+'\n')
    #write only the submisions that have well formed answer as candidates
    with open(infile2, 'r') as f:
        with open(outfile,'w') as w:
            for l in f:
                j = json.loads(l)
                if j['query_id'] in ids:
                    w.write(l)

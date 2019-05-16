import sys
import json
import pandas as pd 

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: tojson.py <input_path> <output_path>')
        exit()
    infile = sys.argv[1]
    outfile = sys.argv[2]
    df = pd.DataFrame()
    with open(infile,'r') as f:
        for l in f:
            j = json.loads(l)
            if 'answers' in j:
                s = pd.Series([j['query'],j['query_id'],j['query_type'],j['passages'],j['answers'],j['wellFormedAnswers']],['query','query_id','query_type','passages','answers','wellFormedAnswers'])
            else:
                s = pd.Series([j['query'],j['query_id'],j['query_type'],j['passages']],['query','query_id','query_type','passages'])
            df = df.append(s, ignore_index = True)
    df.to_json(outfile)

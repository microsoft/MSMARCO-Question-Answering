
#coding=utf8

import sys
import json
import pandas as pd
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: dureader_to_msmarco.py <inputfile> <outputfile>")
        exit()
    else:
        df = pd.DataFrame()
        with open(sys.argv[1],'r') as f:
            for l in f:
                j = json.loads(l)
                j['query_type'] = j.pop('question_type')
                if 'entity_answers' in j:
                    j.pop('entity_answers')
                j.pop('fact_or_opinion')
                
                j['query_id'] = j.pop('question_id')
                j['query'] = j.pop('question')
                passages = []
                for k in j['documents']:
                    data = {}
                    if k['is_selected'] == True:
                        data['is_selected'] = 1
                    else:
                        data['is_selected'] = 0
                    data['passage_text'] = k['paragraphs']
                    data['url'] = ''
                    passages.append(data)
                j['passages'] = passages
                j.pop('documents')
                df = df.append(j,1)
        df.to_json(sys.argv[2])

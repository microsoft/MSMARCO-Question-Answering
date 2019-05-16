import json
import sys
import pandas as pd
YSNO_MAP = set(['is', 'can', 'does', 'are', 'do', 'should', 'did', 'will', 'was', 'has', 'could', 'were', 'would'])

def general_stats_data(path):
    df = pd.read_json(path)
    query_type_label = {'LOCATION': 0, 'DESCRIPTION':0, 'NUMERIC':0, 'ENTITY':0, 'PERSON':0}
    wfa = 0
    total_judgements = 0
    total_wfa_judgments = 0
    multiple_answers = 0
    multiple_wfa = 0
    total_size = len(df)
    for row in df.iterrows():
        category = row[1]['query_type']
        total_judgements += len(row[1]['answers'])
        currentwfa = row[1]['wellFormedAnswers']
        if category in query_type_label:
            query_type_label[category] += 1
        if len(row[1]['answers']) > 1:
            multiple_answers += 1
        if currentwfa != '[]':
            wfa += 1
            total_wfa_judgments += len(currentwfa)
            if len(currentwfa) > 1:
                multiple_wfa += 1
    print('Columns:{}'.format(df.columns.values))
    print('{} queries with {} judgements with {} queries contain more than one judgment'.format(total_size,total_judgements,multiple_answers))
    print('{} queries with Well Formed Answers of which {} contain more than one judgment'.format(wfa,multiple_wfa))
    print('----query distribution by dataset type----')
    for key in query_type_label:
        print(key + ',' + str(query_type_label[key])+ ',' + str(query_type_label[key]/total_size))

def general_stats_data_public(path):
    df = pd.read_json(path)
    query_type_label = {'LOCATION': 0, 'DESCRIPTION':0, 'NUMERIC':0, 'ENTITY':0, 'PERSON':0}
    total_size = len(df)
    for row in df.iterrows():
        category = row[1]['query_type']
        if category in query_type_label:
            query_type_label[category] += 1
    print('Columns:{}'.format(df.columns.values))
    print('{} queries'.format(total_size))
    print('----query distribution by dataset type----')
    for key in query_type_label:
        print(key + ',' + str(query_type_label[key])+ ',' + str(query_type_label[key]/total_size))

def whether_ynq(query):
    tokens = query.split(' ')
    head = tokens[0]
    if head in YSNO_MAP and 'or' not in set(tokens):
        return True
    else: return False

def update_query_type(query, qmap):
    assert len(query) > 0
    query = query.lower()
    head = query.split()[0]
    if whether_ynq(query):
        qmap['yesno'] += 1
    elif head in qmap:
        qmap[head] += 1
    else:
        qmap['other'] +=1

def compute_stat(queries):
    print('----query distribution by wording----')
    query_type_map = {'yesno':0, 'what':0, 'who':0, 'which':0, 'where':0, 'when':0, 'why':0, 'how':0, 'other':0}
    for qid, query in queries.items():
        update_query_type(query, query_type_map)
    total = sum(list(query_type_map.values()))
    print(update_query_type)
    for k, v in query_type_map.items():
        print('{},{},{}'.format(k, v, 1.0 * v/total))
if __name__ == '__main__':
    if len(sys.argv) == 2:
        infile = sys.argv[1]
        general_stats_data(infile)
        with open(infile, 'r', encoding='utf-8') as f:
            inputdata = json.load(f)
        compute_stat(inputdata['query'])
        exit()
    elif len(sys.argv) == 3 and sys.argv[2] == '-p':
        infile = sys.argv[1]
        general_stats_data_public(infile)
        with open(infile, 'r', encoding='utf-8') as f:
            inputdata = json.load(f)
        compute_stat(inputdata['query'])
    else:
        print('Usage: exploredata.py <input_path> <-p for public no answer dataset>')
        exit()

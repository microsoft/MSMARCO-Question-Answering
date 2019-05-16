import sys
import pandas as pd
import matplotlib.pyplot as plt
def get_stats(histogram):
    average = 0.0
    total_values = 0.0
    for value in histogram:
        total_values += histogram[value]
        average += float(histogram[value]) * float(value)
    return average/total_values
def compute_stats(histogram, title):
    average = get_stats(histogram)
    print("###################################\n")
    print("Statistics about {}\n".format(title))
    print("Average length:{}",format(average))
    print("###################################\n")
    plt.bar(list(histogram.keys()), histogram.values(), color='g')
    plt.show()
    for key in sorted(histogram.keys()):
        print("{}:{}".format(key, histogram[key]))
def main():
    file = sys.argv[1]
    df = pd.read_json(file)
    queries = {}
    answers = {}
    well_formed_answers = {}
    passages = {}
    
    for row in df.iterrows():
        queries[row[1]['query']] = 1
        for v in row[1]['answers']:
            answers[v] = 1
        for v in row[1]['wellFormedAnswers']:
            well_formed_answers[v] = 1
        for p in row[1]['passages']:
            passages[p['passage_text']] = 1
    data = {'queries' : queries, 'answers' : answers, 'well_formed_answers' : well_formed_answers, 'passages' : passages}
    for value in data:

        histogram = {}
        for v in data[value]:
            l = len(v.split())
            if l in histogram:
                histogram[l] += 1
            else:
                histogram[l] = 1
        compute_stats(histogram, value)
if __name__ == '__main__':
    main()

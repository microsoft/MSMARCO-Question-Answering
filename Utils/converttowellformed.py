import pandas as pd
import sys

def makewf(input,output):
    df = pd.read_json(input)
    df = df.drop('answers',1)
    df = df.rename(columns={'wellFormedAnswers':'answers'})
    df = df[df.answers != '[]']
    df.to_json(output)
    return

if __name__ == '__main__':
    if len(sys.argv) == 3:
        makewf(sys.argv[1],sys.argv[2])
    else:
        print("Usage: converttowellformed.py <input file> <desired outputname>")

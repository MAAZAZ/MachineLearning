import numpy as np
import pandas as pd
from fractions import Fraction
from collections import Counter
import functools 

def classify(file, inputData):
    data = pd.read_csv(file)
    data=np.array(data)
    data=data[1:,:]
    # target
    classes = [t[-1] for t in data]
    # how mush data for each classes
    class_counts = dict(Counter(classes))
    # get classes
    classes = list(set(classes))
    # probablity for each classes
    class_probs = dict([(c,Fraction(class_counts[c])/len(data)) for c in classes])
    # 
    print([t[:len(t)-1] for t in data])
    #print([(c, np.transpose([t[:len(t)-1] for t in data if t[-1] == c])) for c in classes])
    classified_data = [(c, np.transpose([t[:len(t)-1] for t in data if t[-1] == c])) for c in classes]
    classified_counts = dict([(c[0], [dict(Counter(a)) for a in c[1]]) for c in classified_data])

    results = []
    for c in classes:
        probs = [Fraction(v.get(k, 0))/class_counts[c] for k,v in zip(inputData, classified_counts[c])]
        prob = functools.reduce(lambda x,y: x*y, probs) * class_probs[c]
        results.append((prob, c))
    return max(results)


if __name__ == "__main__":
    i1 = [11,33,7,136,5,5,0,1,1,4.4,20.75,0,15.25,35.7,42,211.45,125,1,1,0,1,1,0,1.482,3.033,4.913,4]
    print("predict de i1 ==>"+ str(classify("ChurnData.csv", i1)))

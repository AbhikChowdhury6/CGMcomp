import pandas as pd
import pickle
import csv
import numpy as np

#generate features function
def extractFeatures(sglist):
    features = []
    #min max diffrence
    #features.append(max(sglist) - min(sglist))
    #time between min and max
    #features.append(sglist.index(max(sglist)) - sglist.index(min(sglist)))
    #start - end
    features.append((sglist[0]-sglist[23]))
    
    #fft
    #features.append(sum(abs(np.fft.fft(sglist)))/len(sglist))
        
    #integrate
    integrated = []
    for i in range(23):
        integrated.append(sglist[i]-sglist[i+1])
    
    integratedSquared = [x ** 2 for x in integrated]

    #max speed
    features.append(max(integrated))
    
    #average velocity
    features.append(sum(integrated)/len(integrated))
    
    #average squared velocity 
    features.append(sum(integratedSquared)/len(integratedSquared))
    
    return features


#load classifier
loaded_model = pickle.load(open("model.pickle", 'rb'))


#run classifier
with open('test.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    results = []

    testsMatrix = []
    for row in readCSV:
        numbers = [float(i) for i in row]
        #print(numbers)
        feats = extractFeatures(numbers)
        #print(feats)
        testsMatrix.append(feats)
        
    results = loaded_model.predict(testsMatrix)
    print(results)
    #save results to file
    with open('Results.csv', 'w') as f: 
        write = csv.writer(f) 
        for r in results.tolist():
            write.writerow([r])


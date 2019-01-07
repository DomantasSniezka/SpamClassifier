import os
import io
import numpy as np
from pandas import DataFrame

from sklearn.feature_extraction.text import CountVectorizer # operate on lots of words at once
from sklearn.naive_bayes import MultinomialNB # Naive Bayes


# Function that iterates trough all files in a directory
def readFiles(path):
    for root, dirnames, filenames in os.walk(path) :
        for fileName in filenames:
            path = os.path.join(root, fileName)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n' :
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


# Function to retrieve data from directory
def dataFrameFromDirectory(path, classification ) :
    rows = []
    index = []

    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification })
        index.append(filename)

    return DataFrame(rows, index=index)



data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('C:\\Users\\doman\\Documents\\Data Science\\DataScience-Python3\\emails\\spam', 'spam'))
data = data.append(dataFrameFromDirectory('C:\\Users\\doman\\Documents\\Data Science\\DataScience-Python3\\emails\\valid', 'valid'))

# preview DataFrame
print(data.head())


# counting number of words in each email message

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)


# Examples to test vectorizer and classifier
examples = ["Free Cash!! enter Now!!", "Jimmy How are you?"]
examples_count = vectorizer.transform(examples)

predictions = classifier.predict(examples_count)
print("\n\nThe Emails given are: \n", predictions)

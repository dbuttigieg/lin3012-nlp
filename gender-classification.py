import glob

import pandas as pd

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from textblob import TextBlob


# assign labels according to file name.
# check if filename has '.male.' and '.female.' for accuracy
def get_data_label(filename):
    if filename.__contains__(".male."):
        label = "male"
    if filename.__contains__(".female."):
        label = "female"
    return label


# function to eliminate blank lines from data
def non_blank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


# extract text from blogs and get gender, given a directory
def get_data_from_file(path_to_data):
    data_folder = glob.glob(path_to_data)
    data_set = []

    # iterate on every document
    for blog_file in data_folder:
        label = get_data_label(blog_file)

        # open file with read access
        f = open(blog_file, 'r')
        lines = []

        # read the document line by line, eliminating blank lines
        for c in non_blank_lines(f.readlines()):
            # remove tags and unnecessary annotations from the document
            if c.startswith('<date>') \
                    or c.startswith('<Blog>') or c.startswith('</Blog>') \
                    or c.startswith('<post>') or c.startswith('</post>'):
                c = ''
            else:
                # append all the valid lines to a List
                lines.append(c)

        # take the List of lines, and convert them into a string. eliminate delimeters
        string_content = ''.join(lines)
        string_content.replace('\r', '')
        string_content.replace('\n', '')
        string_content.replace('\t', '')
        string_content.rstrip()
        string_content = unicode(string_content, errors='ignore')
        # add the item to a List
        data_set.append({'data': string_content, 'gender': label})

        f.close()

    # add the whole List of extracted documents to the DataFrame
    data = pd.DataFrame(data_set)
    return data


# get the base form of the words - normalisation
def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]


# the data is stored into a Data Frame, provided by the pandas library
# this data structure enables the data to be stored in a table-like format with labelled columns
my_data = pd.DataFrame({'data': [], 'gender': []})

# populate the DataFrame
my_data = my_data.append(get_data_from_file('./training-data/*'))

# split training and testing data, reserving 20% of the data for testing
train_data, test_data = train_test_split(my_data, test_size=0.2)


# CLASSIFICATION
#
# Build a Pipeline to perform feature extraction and training on the data
# extract features into a bag-of-words model using a CountVectorizer - get the number of times a word occurs in a doc
# normalize the frequency using tf-idf to make the data more accurate, reflecting the context
# then train and predict using three different approaces
# 1. Naive Bayes Classifier
# 2. SVM with Stratified 5-Fold cross validation
# 3. SVM with cross validation, cv=5.

# APPROACH 1: NAIVE BAYES
naive_bayes_pipeline = Pipeline([
    ('bow_transformer', CountVectorizer(analyzer=split_into_lemmas, stop_words='english')), # tokenize
    ('tf_idf', TfidfTransformer()), # normalize
    ('classifier', MultinomialNB()) # train and predict
])

# output results from Naive Bayes Classifier
naive_bayes = naive_bayes_pipeline.fit(train_data['data'], train_data['gender'])
predictions_nb = naive_bayes.predict(test_data['data'])
print "TESTING USING MULTINOMIAL NAIVE BAYES"
print "Expected Output: ", test_data['gender']
print "Actual Output: ", predictions_nb
# output a classification report, evaluating the accuracy, precision, recall, and f1 score for the results
print "Classification Report: "
print classification_report(test_data['gender'], predictions_nb)


# APPROACH 2: SVC with Stratified K-Fold Cross Validation
svm_pipeline = Pipeline([
    ('bow_transformer', CountVectorizer(analyzer=split_into_lemmas, stop_words='english')),
    ('tf_idf', TfidfTransformer()),
    ('classifier', SVC())
])

# SVM parameters for pipeline so we could tune automatically
svm_params = {'classifier__C': [1], 'classifier__gamma': [0.001], 'classifier__kernel': ['linear']}

grid_svm_skf = GridSearchCV(
    svm_pipeline,  # pipeline from above
    param_grid=svm_params,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=1,  # number of cores to use for parallelization
    scoring='accuracy',
    cv=StratifiedKFold(train_data['gender'], n_folds=5),  # using StratifiedKFold CV with 5 folds
)

# evaluate results
svm_skf = grid_svm_skf.fit(train_data['data'], train_data['gender'])
predictions_svm_skf = svm_skf.predict(test_data['data'])
print "TESTING USING SVM WITH STRATIFIED 5-FOLD CROSS VALIDATION"
print "Expected Output: ", test_data['gender']
print "Actual Output: ", predictions_svm_skf
print "Classification Report: "
print classification_report(test_data['gender'], predictions_svm_skf)

# APPROACH 3: CROSS VALIDATION = 5
grid_svm_cv_5 = GridSearchCV(
    svm_pipeline,
    param_grid=svm_params,
    refit=True,
    n_jobs=1,
    scoring='accuracy',
    cv=5,  # cross validation equal 5 will compute the scores 5 times with different splits
)

# evaluate results
svm_cv_5 = grid_svm_cv_5.fit(train_data['data'], train_data['gender'])
predictions_cv_5 = svm_cv_5.predict(test_data['data'])
print "TESTING USING SVM WITH CV = 5"
print "Expected Output: ", test_data['gender']
print "Actual Output: ", predictions_cv_5
print "Classification Report: "
print classification_report(test_data['gender'], predictions_cv_5)

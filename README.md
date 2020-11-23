# Text-classification-practice
Linear models in comment classification

In this project we used our own implementations of logistic regression for binary classifications, gradient descent and
stochastic gradient descent. 
The goal was to determine toxic comments in Wikipedia discussion page.

We used such tricks to work with text features:
* casting to lower case
* text processing with regular expressions (re library)
* lemmatization with part of speech recognition (ntlk library)
* stop words filtration

For text feature encoding were used Bag of Words and TF-IDF methods.

Initial task description located in file "task.pdf".

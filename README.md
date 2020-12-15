# Comparing-ML-For-DDoS

The project is a comparison of 3 different ML classifiers in their prediction accuracy and speed of DDoS attack conditions. 

# The classifiers 

1) OneVsRestClassifier
2) MultinomialNB
3) BaggingClassifier 

After using these simple algorithms, they are inturn compared with ensemble algorithms taken a combination of the above 3 mentioned algorithms

# The ensemble classifiers 

1) Voting Classifier 
2) Weighted Classifier 
3) Stacking Classifier

3 permutations of the weights of the base classifiers are taken for 3 readings of the metrics of the ensemble classifiers
# Metrics used for comparison

Accuracy(%)
Mean 
Standard Deviation
Precision
Recall 
F-Measure
True Postive
True Negative
False Positive

# Datasets used

I did not get the permission to upload datasets, five datasets were used each containing network data and DDoS conditions for one kind of protocol 
The protocols are: ICMP, LAND , TCPSYN, TCPSYNACK , UDP

# Detecting-Phishing-Websites-using-ML

Phishing is a type of social engineering where an attacker sends a fraudulent (e.g., spoofed, fake, 
or otherwise deceptive) message designed to trick a human victim into revealing sensitive 
information to the attacker or to deploy malicious software on the victim's infrastructure like 
ransomware. Phishing attacks have become increasingly sophisticated and often transparently 
mirror the site being targeted, allowing the attacker to observe everything while the victim is 
navigating the site, and transverse any additional security boundaries with the victim.


 Detecting Phishing Websites Using Machine Learning
1. Introduction
Phishing is a type of social engineering where an attacker sends a fraudulent (e.g., spoofed, fake, 
or otherwise deceptive) message designed to trick a human victim into revealing sensitive 
information to the attacker or to deploy malicious software on the victim's infrastructure like 
ransomware. Phishing attacks have become increasingly sophisticated and often transparently 
mirror the site being targeted, allowing the attacker to observe everything while the victim is 
navigating the site, and transverse any additional security boundaries with the victim.
It is a method of identity theft that relies on individuals unwittingly volunteering personal details 
or information that can then be used for nefarious purposes. It is often carried out through the 
creation of a fraudulent website, email, or text appearing to represent a legitimate firm.
It involves compromising legitimate web pages in order to redirect users to a malicious website 
or an exploit kit via cross-site scripting. A hacker may compromise a website and insert an 
exploit kit such as MPack in order to compromise legitimate users who visit the now 
compromised web server. One of the simplest forms of page hijacking involves altering a 
webpage to contain a malicious inline frame which can allow an exploit kit to load. Page 
hijacking is frequently used in tandem with a watering hole attack on corporate entities in order 
to compromise targets.
A scammer may use a fraudulent website that appears on the surface to look the same as a 
legitimate website. Visitors to the site, thinking they are interacting with a real business, may 
submit their personal information, such as social security numbers, account numbers, login IDs, 
and passwords, to this site. The scammers then use the information submitted to steal visitors' 
money, identity, or both; or to sell the information to other criminal parties.
Malicious links will lead to a website that often steals login credentials or financial information 
like credit card numbers. Attachments from phishing emails can contain malware that once 
opened can leave the door open to the attacker to perform malicious behavior from the user’s 
computer.
The importance of safeguarding online users from becoming victims of online fraud, divulging 
confidential information to an attacker among other effective uses of phishing as an attacker’s 
tool, phishing detection tools play a vital role in ensuring a secure online experience for users. 
Unfortunately, many of the existing phishing-detection tools, especially those that depend on an 
existing blacklist, suffer limitations such as low detection accuracy and high false alarm that is 
often caused by either a delay in blacklist update as a result of the human verification process 
involved in classification or perhaps, it can be attributed to human error in classification which 
may lead to improper classification of the classes.
2. Dataset Overview
The Data I am using consists of Phishing Websites and benign websites. The Phishing Websites 
Data Set was collected mainly from the PhishTank archive, MillerSmiles archive, Googleâ€™s 
searching operators. The benign websites were collected from Alexa.com.
The algorithms used on this dataset have never been used on it. The few other projects done are 
on different datasets with not many attributes, I even found out missing attributes from other 
datasets that were very important to the decision making. It’s a new data with new attributes, I 
believe my contributions are using this data and using the classifier algorithms and post and pre 
pruning the data to avoid overfitting was one of the challenges. Other projects and literature 
reviews I have read have not applied this classification problems to this kind of data, with this 
many attributes, they have not used this new and updated dataset and have not applied Pre￾Pruning and Post Pruning. So that was a first and it really made the data easy to apply my models 
onto it.
There are a lot of algorithms and a wide variety of data types of phishing detection out there. A 
phishing URL and the corresponding page have several features which can be differentiated from 
a malicious URL. The data I have chosen has 11,055 instances and 32 attributes.based on certain 
features: 
1. Address-based features:
● IP Address in URL
● Length of URL
● Using URL shortening services like 
“TinyURL”
● URL’s having “@” symbol
● Redirecting using “//”
● Sub Domain and Multi-Sub Domains
● HTTPS in Domain name
● Domain Registration Length
● Favicon
● Using Non-Standard Port
● The existence of “HTTPS” token in 
the domain part of the URL
● Adding prefix or suffix separated by 
(-) to the Domain
2. Abnormal-based Features:
● Request URL
● URL of anchor
● Link in <Meta>, <Script> and 
<Link> tags
● Server Form Handler (SFH)
● Submitting information to an email
● Abnormal URL
3. HTML and JavaScript-based features:
● Website Forwarding
● Status bar customization
● Disabling right-click
● Using pop-up window
● IFrame redirection
4. Domain-based features:
● Age of domain
● DNS record
● Website traffic
● PageRank
● Google index
● Number of links pointing to the 
page
● Statistical reports based feature
● There are in total 28 features that were selected from the dataset for this model.
3. Loading the Data
The first step is to load the data and see what the first few instances and some of the columns are
Figure 1: Loading the data
Figure 2: Loading the data output
4. Familiarizing with the Data
Now that we have loaded the data, we try to look into the data and get familiar with it. Gain 
information about the data, if it has any missing values; all the data types of the attributes are the 
same, and get the shape of the data. 
Information about the data and its attribute types.
Figure 3: Display info
Figure 4
The shape of the data:
Figure 5: Shape of the data
Displaying all the columns in the dataset
Figure 6: Shape of the data output
Displaying all attributes values data type to see if there are types that need to be changed.
Figure 7: Display dtypes
In this data all the values are of type int64 there are no categorical values that need to be 
changed.
5. Visualizing the Data
The next step is to visualize the data and see if there is a clear idea of what the information 
means by giving it visual context through maps or graphs. This makes it easier for us to 
comprehend and therefore makes it easier to identify trends, patterns, and outliers within large 
data sets. Plots and graphs like histograms and heatmaps are displayed to find how the data is 
distributed and how the features are related to each other.
● Histogram: Plotting the histogram of the data to summarise discrete or continuous data 
that are measured on an interval scale and to illustrate the major features of the 
distribution of the data in a convenient form.
Figure 8: Histogram
● Heatmap: Plotting the heatmap for the data to better visualize the volume of 
locations/events within a dataset and show relationships and correlation between the 
attributes.
Figure 9: Heatmap
6. Data Preprocessing and EDA
The next step is to clean and use data preprocessing techniques and transform the data to use it in 
the models. The data may have many irrelevant and missing parts. To handle it we need to find 
out if the dataset has missing values and noisy data; if it needs to be normalized or discretized; if 
the data needs to be reduced; or if there are any continuous variables.
Figure 10: Describe
We use “describe” here to see the count, mean, std, minimum values, quartiles (25%, 50%, 
75%), and maximum values for all the attributes. 
From this, we learned that most of the data is made of {-1,1} values except for “id” which is the 
number of instances, and we are gonna drop that attribute cause it has no significance to our ML 
learning model. 
Figure 11: Dropping Id
We dropped the “id” attribute and its values and assigned it to a new dataset variable, called 
“dataset1”.
Next, we check if the data has any null or missing values and we do that by:
Figure 12: isnull()
Output:
Figure 13: isnull() output
From the above figure, we can see the data doesn’t have any null or missing values. Now the 
data is ready to be trained.
7. Splitting the Data
Here we split the data into training and test data 80-20 for our models to use. We first assign the 
class “Result”, into a variable to compare when we apply the models, and then we drop the class 
from our variable that’s gonna be split into training and test data.
Figure 14:Splitting the data
We used train_test_split to split the dataset, while also shuffling the data, i.e the training and test 
data are randomly selected
Figure 15
We randomly assign the data into the train and test sets.
Output:
Figure 16
8. Machine Learning Models and Training Algorithm
This is a supervised machine learning task, so Classification and Regression are the options to 
use. This data set comes under a classification problem, as the input URL is classified as 
phishing or legitimate. The classification techniques considered to train the data are:
● Decision Tree Algorithm
● Random Forest Algorithm
● Support Vector Machine Algorithm
a. Decision Tree Algorithm
A Decision Tree is a supervised machine learning algorithm that can be used for both Regression 
and Classification problem statements. It divides the complete dataset into smaller subsets while 
at the same time an associated Decision Tree is incrementally developed. The data is 
continuously split according to a certain parameter. The tree can be explained by two entities, 
namely decision nodes and leaves. The leaves are the decisions or the final outcomes. And the 
decision nodes are where the data is split.
In a Decision Tree Algorithm, the first step is to split the data into training and test, which was 
already done above. Next step is to use the classifier algorithm on the training sets without any 
parameters and see the results.
Figure 17
Output:
Figure 18
Displayed are the confusion matrix and the classification report, which consists of accuracy, 
recall, f1-score, and support. 
The model’s accuracy on training data is 99% and the accuracy for the test data is 95.7%. By 
looking at this report it can be seen the model overfits for the data since the train data accuracy is 
bigger than the test data accuracy.
Next step is to apply pre pruning, to determine the maximum depth the tree can go, the minimum 
samples split, and the minimum samples leaf.
Pre Pruning:
Figure 19: Pre-Pruning
Output:
Figure 20: Pre-Pruning output
The Pre-Pruning result outputs the best values of max_depth, min_samples_leaf, and the 
min_samples_split. These are values set to stop the tree from overfitting. 
The Pre-Pruning accuracy score after putting the values from the above figure are:
Figure 21
Output:
Figure 22: max_depth
From the above figure, it is seen that the model has improved but still needs more values to 
improve the accuracy for the test data.
The next step is to post prune it and find the best ccp_alpha, Cost complexity pruning. Cost 
complexity pruning provides another option to control the size of a tree. In 
DecisionTreeClassifier, this pruning technique is parameterized by the cost complexity 
parameter, ccp_alpha. Greater values of ccp_alpha increase the number of nodes pruned.
To find the best value of ccp_alpha, it is required to plot the AUC-ROC score vs the alpha value 
to see where the accuracy is maximum for both training and test datas.
Figure 23: ccp_alpha
Output:
Figure 23: ccp_alpha to accuracy graph
From the above figure the maximum accuracy the model can reach is when the alpha value is at 
0.001. Now combining all the values gives us an output of:
Figure 24: ccp_alpha evaluation
Figure 25
Performance Evaluation:
Figure 26: Performance Evaluation
Now the accuracy for the test data is greater than the accuracy for the training data, which 
significantly increases the model’s capability of classifying new data.
Visualizing the tree:
Figure 27: Plotting the Decision Tree
Output:
Figure 28: Decision Tree
This is what the tree looks like, as it can be seen the maximum depth it has undergone is 10 and 
the root node in the tree looks like the figure below:
Figure 29: Root Node
The last step is to check the feature importance in the model:
Figure 30: Plotting Feature Importance
Output:
Figure 31: Feature Importance
b. Random Forest Algorithm
Random forest is a flexible, easy-to-use machine learning algorithm that produces, even without 
hyper-parameter tuning, a great result most of the time. It is also one of the most used 
algorithms, because of its simplicity and diversity (it can be used for both classification and 
regression tasks). It is a supervised learning algorithm. The "forest" it builds is an ensemble of 
decision trees, usually trained with the “bagging” method. The general idea of the bagging
method is that a combination of learning models increases the overall result.
The first step for Random Forest is to determine the best possible values of its parameters, i.e. 
n_estimators (This is the number of trees you want to build before taking the maximum voting or 
averages of predictions. A higher number of trees give you better performance but makes your 
code slower.), max_depth, and min_samples_leaf.
To determine the best value of n_estimators we use GridSearchCv to plot the AUC_Score with 
respect to n_estimators.
Figure 32: n_estimators
Output:
Figure 33: n_estimators to AUC-Score
The next step is to find out the best value for max_depth. That is also done using GridSearchCv 
in the following way:
Figure 34: max_depths
Figure 35: max_depths to AUC Score
Now that we know the best values for max_depth and n_estimator, we can use the Random 
Forest Classifier to use those values and predict the target values.
Figure 36: Random Forest Classifier
Figure 37
Performance Evaluation:
Figure 38: Random Forest Classifier Performance Evaluation
Plotting a Random Tree is computationally costly cause it computes and calculates 100 trees:
Figure 39: Plotting Random Forest
Output:
The tree is big and takes quite a long time to run, this is some of the root nodes on the tree:
Figure 40: Random Forest Tree
Figure 41: Random Forest Tree sample
The last step is to check the feature importance in the model:
Figure 42: Plotting Random Forest Feature Importance
Output:
Figure 43: Random Forest Feature Importance
c. SVM Algorithm
In machine learning, support vector machines (SVMs, also support vector networks) are 
supervised learning models with associated learning algorithms that analyze data used for 
classification and regression analysis. Given a set of training examples, each marked as 
belonging to one or the other of two categories, an SVM training algorithm builds a model that 
assigns new examples to one category or the other, making it a non-probabilistic binary linear 
classifier.
Figure 44: svm.SVC
After we fit the model, we are ready to use it to predict the test data.
Figure 45: SVM accuracy score
Performance Evaluation:
Figure 46: SVM performance evaluation
9. Comparison of Models 
To compare the models’ performance, a data frame is created. The columns of this data frame are 
the lists created to store the results of the model.
Figure 47: Comparisons of Model
The models are evaluated, and the considered metric is accuracy. From the above figures, it is 
shown that Random Forest gives better performance. The last step is to save and load the model 
and test out the model on some sample data.
Figure 48: Loading and testing Model
Predicting the data:
Figure 49: Predicting Result
This test was conducted on an 80-20 split, further splitting testing was done. The comparison of 
the models when the data is split on an 80-20, 60-40, and 50-50 are:
Figure 50: 80-20 split data performance evaluation Figure 51: 60-40 split data performance evaluation
Figure 52: 50-50 split data performance evaluation
Table1: Comparison of Models
In all cases, the Random Forest algorithm performs much better than the rest.
10. Conclusion
This project aims to enhance detection methods to detect phishing websites using machine 
learning technology. Detection accuracy of 97.0% was achieved using the Random Forest 
algorithm with the lowest false positive rate. The result shows that classifiers give better 
performance when more instances and more attributes was used. 
The problem of phishing cannot be eradicated, nonetheless can be reduced by combating it in 
two ways, improving targeted anti-phishing procedures and techniques and informing the public 
on how fraudulent phishing websites can be detected and identified. To combat the ever evolving
and complexity of phishing attacks and tactics, ML anti-phishing techniques are essential.
The outcome of this project reveals that the proposed method presents superior results rather than 
the existing deep learning methods. It has achieved better accuracy and F1—score with a limited 
amount of time. The future direction of this project is to develop an unsupervised deep learning 
method to generate insight from a URL.
The algorithms used on this dataset have never been used on it. The few other projects done are 
on different datasets with not many attributes, I even found out missing attributes from other 
datasets that were very important to the decision making. It’s a new data with new attributes, I 
believe my contributions are using this data and using the classifier algorithms and post and pre 
pruning the data to avoid overfitting was one of the challenges. Other projects and literature 
reviews I have read have not applied this classification problems to this kind of data, with this 
many attributes, they have not used this new and updated dataset and have not applied Pre￾Pruning and Post Pruning. So that was a first and it really made the data easy to apply my models 
onto it. 
Working on this project was very knowledgeable and worth the effort. In the future, this project 
could be used as a browser extension or an application with a user interface that users can use to 
detect phishing websites more easily.
 
 ![image](https://user-images.githubusercontent.com/33021726/149592001-33c1df0a-ce0b-467b-9a57-5e75c7a735f8.png)


# Detecting-Phishing-Websites-using-ML
Detecting Phishing Websites Using Machine Learning


1.	Introduction

Phishing is a type of social engineering where an attacker sends a fraudulent (e.g., spoofed, fake, or otherwise deceptive) message designed to trick a human victim into revealing sensitive information to the attacker or to deploy malicious software on the victim's infrastructure like ransomware. Phishing attacks have become increasingly sophisticated and often transparently mirror the site being targeted, allowing the attacker to observe everything while the victim is navigating the site, and transverse any additional security boundaries with the victim.
It is a method of identity theft that relies on individuals unwittingly volunteering personal details or information that can then be used for nefarious purposes. It is often carried out through the creation of a fraudulent website, email, or text appearing to represent a legitimate firm.
It involves compromising legitimate web pages in order to redirect users to a malicious website or an exploit kit via cross-site scripting. A hacker may compromise a website and insert an exploit kit such as MPack in order to compromise legitimate users who visit the now compromised web server. One of the simplest forms of page hijacking involves altering a webpage to contain a malicious inline frame which can allow an exploit kit to load. Page hijacking is frequently used in tandem with a watering hole attack on corporate entities in order to compromise targets.
A scammer may use a fraudulent website that appears on the surface to look the same as a legitimate website. Visitors to the site, thinking they are interacting with a real business, may submit their personal information, such as social security numbers, account numbers, login IDs, and passwords, to this site. The scammers then use the information submitted to steal visitors' money, identity, or both; or to sell the information to other criminal parties.
Malicious links will lead to a website that often steals login credentials or financial information like credit card numbers. Attachments from phishing emails can contain malware that once opened can leave the door open to the attacker to perform malicious behavior from the user’s computer.
The importance of safeguarding online users from becoming victims of online fraud, divulging confidential information to an attacker among other effective uses of phishing as an attacker’s tool, phishing detection tools play a vital role in ensuring a secure online experience for users. Unfortunately, many of the existing phishing-detection tools, especially those that depend on an existing blacklist, suffer limitations such as low detection accuracy and high false alarm that is often caused by either a delay in blacklist update as a result of the human verification process involved in classification or perhaps, it can be attributed to human error in classification which may lead to improper classification of the classes.
2.	Dataset Overview
The Data I am using consists of Phishing Websites and benign websites. The Phishing Websites Data Set was collected mainly from the PhishTank archive, MillerSmiles archive, Googleâ€™s searching operators. The benign websites were collected from Alexa.com.
The algorithms used on this dataset have never been used on it. The few other projects done are on different datasets with not many attributes, I even found out missing attributes from other datasets that were very important to the decision making. It’s a new data with new attributes, I believe my contributions are using this data and using the classifier algorithms and post and pre pruning the data to avoid overfitting was one of the challenges. Other projects and literature reviews I have read have not applied this classification problems to this kind of data, with this many attributes, they have not used this new and updated dataset and have not applied Pre-Pruning and Post Pruning. So that was a first and it really made the data easy to apply my models onto it. 
 
There are a lot of algorithms and a wide variety of data types of phishing detection out there. A phishing URL and the corresponding page have several features which can be differentiated from a malicious URL. The data I have chosen has 11,055 instances and 32 attributes.based on certain features: 
1.	Address-based features:
●	IP Address in URL
●	Length of URL
●	Using URL shortening services like “TinyURL”
●	URL’s having “@” symbol
●	Redirecting using “//”
●	Adding prefix or suffix separated by (-) to the Domain
	●	Sub Domain and Multi-Sub Domains
●	HTTPS in Domain name
●	Domain Registration Length
●	Favicon
●	Using Non-Standard Port
●	The existence of “HTTPS” token in the domain part of the URL

2.	Abnormal-based Features:
●	Request URL
●	URL of anchor
●	Link in <Meta>, <Script> and <Link> tags	●	Server Form Handler (SFH)
●	Submitting information to an email
●	Abnormal URL

3.	HTML and JavaScript-based features:
●	Website Forwarding
●	Status bar customization
●	Disabling right-click	●	Using pop-up window
●	IFrame redirection

4.	Domain-based features:
●	Age of domain
●	DNS record
●	Website traffic
●	PageRank	●	Google index
●	Number of links pointing to the page
●	Statistical reports based feature
●	There are in total 28 features that were selected from the dataset for this model.


3.	Loading the Data

The first step is to load the data and see what the first few instances and some of the columns are
![image](https://user-images.githubusercontent.com/33021726/149592734-ca4ce486-af48-4e40-aaa4-a0ece159b7af.png)
Figure 1: Loading the data
 
![image](https://user-images.githubusercontent.com/33021726/149592797-1afb8375-2863-40f9-835d-a850f55c5253.png)
Figure 2: Loading the data output

4.	Familiarizing with the Data

Now that we have loaded the data, we try to look into the data and get familiar with it. Gain information about the data, if it has any missing values; all the data types of the attributes are the same, and get the shape of the data. 

Information about the data and its attribute types.
![image](https://user-images.githubusercontent.com/33021726/149592815-7c950018-b0ba-41e4-8302-41ceb5b5459b.png)
Figure 3: Display info


![image](https://user-images.githubusercontent.com/33021726/149592852-6c99c3a6-36a4-43fd-af0a-b0ebde1795e9.png)
Figure 4






The shape of the data:
![image](https://user-images.githubusercontent.com/33021726/149592893-33059486-7dd5-4dfc-81b1-8850b6c7b45c.png)
 
Figure 5: Shape of the data

Displaying all the columns in the dataset
 ![image](https://user-images.githubusercontent.com/33021726/149592924-5821c486-6ca6-4dda-9a5a-3f1f345f4840.png)

 Figure 6: Shape of the data output

Displaying all attributes values data type to see if there are types that need to be changed.
![image](https://user-images.githubusercontent.com/33021726/149592950-a2e244a8-6161-4d2c-a1d7-b6760a1c6848.png)

Figure 7: Display dtypes

In this data all the values are of type int64 there are no categorical values that need to be changed.


5.	Visualizing the Data

The next step is to visualize the data and see if there is a clear idea of what the information means by giving it visual context through maps or graphs. This makes it easier for us to comprehend and therefore makes it easier to identify trends, patterns, and outliers within large data sets. Plots and graphs like histograms and heatmaps are displayed to find how the data is distributed and how the features are related to each other.

●	Histogram: Plotting the histogram of the data to summarise discrete or continuous data that are measured on an interval scale and to illustrate the major features of the distribution of the data in a convenient form.

	![image](https://user-images.githubusercontent.com/33021726/149592966-c3a842bb-76cf-477a-8658-bfa33a536525.png)

Figure 8: Histogram

●	Heatmap: Plotting the heatmap for the data to better visualize the volume of locations/events within a dataset and show relationships and correlation between the attributes.
 ![image](https://user-images.githubusercontent.com/33021726/149592979-5f36df3a-977e-41e9-8cbd-d3032f90ce03.png)

Figure 9: Heatmap


6.	Data Preprocessing and EDA

The next step is to clean and use data preprocessing techniques and transform the data to use it in the models. The data may have many irrelevant and missing parts. To handle it we need to find out if the dataset has missing values and noisy data; if it needs to be normalized or discretized; if the data needs to be reduced; or if there are any continuous variables.

![image](https://user-images.githubusercontent.com/33021726/149592991-1102bcd8-c9a7-43ee-bf3e-d6cd3abda6ab.png)

Figure 10: Describe


We use “describe” here to see the count, mean, std, minimum values, quartiles (25%, 50%, 75%), and maximum values for all the attributes. 

From this, we learned that most of the data is made of {-1,1} values except for “id” which is the number of instances, and we are gonna drop that attribute cause it has no significance to our ML learning model. 

 ![image](https://user-images.githubusercontent.com/33021726/149593005-7072add7-3583-4346-b569-7ba2edd976db.png)

Figure 11: Dropping Id

We dropped the “id” attribute and its values and assigned it to a new dataset variable, called “dataset1”.

Next, we check if the data has any null or missing values and we do that by:
 ![image](https://user-images.githubusercontent.com/33021726/149593010-67566de3-58d8-44aa-b948-b7fff6c4750a.png)

Figure 12: isnull()

Output:
![image](https://user-images.githubusercontent.com/33021726/149593019-196f521b-769f-42cc-b3cb-99fde556a200.png)

Figure 13: isnull() output

From the above figure, we can see the data doesn’t have any null or missing values. Now the data is ready to be trained.

7.	Splitting the Data
Here we split the data into training and test data 80-20 for our models to use. We first assign the class “Result”, into a variable to compare when we apply the models, and then we drop the class from our variable that’s gonna be split into training and test data.

 ![image](https://user-images.githubusercontent.com/33021726/149593036-1590149e-1b3b-4f59-aab1-8a80f1a2e275.png)

Figure 14:Splitting the data

We used train_test_split to split the dataset, while also shuffling the data, i.e the training and test data are randomly selected
 ![image](https://user-images.githubusercontent.com/33021726/149593042-eee5865f-e874-4215-a3d0-af9a761eb922.png)

Figure 15

We randomly assign the data into the train and test sets.
Output:
 ![image](https://user-images.githubusercontent.com/33021726/149593049-ca62f7b4-5e24-4f94-9d26-efdb7c6e32ac.png)

Figure 16


8.	Machine Learning Models and Training Algorithm
This is a supervised machine learning task, so Classification and Regression are the options to use. This data set comes under a classification problem, as the input URL is classified as phishing or legitimate. The classification techniques considered to train the data are:

●	Decision Tree Algorithm
●	Random Forest Algorithm
●	Support Vector Machine Algorithm




a.	Decision Tree Algorithm

A Decision Tree is a supervised machine learning algorithm that can be used for both Regression and Classification problem statements. It divides the complete dataset into smaller subsets while at the same time an associated Decision Tree is incrementally developed. The data is continuously split according to a certain parameter. The tree can be explained by two entities, namely decision nodes and leaves. The leaves are the decisions or the final outcomes. And the decision nodes are where the data is split.

In a Decision Tree Algorithm, the first step is to split the data into training and test, which was already done above. Next step is to use the classifier algorithm on the training sets without any parameters and see the results.

 ![image](https://user-images.githubusercontent.com/33021726/149593055-1eae4902-155c-4e63-a53a-602b6c4b8093.png)

Figure 17


Output:
 ![image](https://user-images.githubusercontent.com/33021726/149593102-f8625343-9025-4696-bd19-910103363f20.png)

Figure 18

Displayed are the confusion matrix and the classification report, which consists of accuracy, recall, f1-score, and support. 

The model’s accuracy on training data is 99% and the accuracy for the test data is 95.7%. By looking at this report it can be seen the model overfits for the data since the train data accuracy is bigger than the test data accuracy.

Next step is to apply pre pruning, to determine the maximum depth the tree can go, the minimum samples split, and the minimum samples leaf.

Pre Pruning:
 ![image](https://user-images.githubusercontent.com/33021726/149593111-3ae63098-54b5-4e6f-836e-eaf0a0853cc9.png)

Figure 19: Pre-Pruning


Output:
 ![image](https://user-images.githubusercontent.com/33021726/149593123-4ee21c4c-b6b0-4ebe-a835-245f524084b0.png)

Figure 20: Pre-Pruning output


The Pre-Pruning result outputs the best values of max_depth, min_samples_leaf, and the min_samples_split. These are values set to stop the tree from overfitting. 

The Pre-Pruning accuracy score after putting the values from the above figure are:
![image](https://user-images.githubusercontent.com/33021726/149593134-3addee1b-c0a5-405d-afcd-bae96e785280.png)
 
Figure 21

Output:
 ![image](https://user-images.githubusercontent.com/33021726/149593143-8cc9330f-8e5e-4fdd-9d64-6f944389f490.png)

Figure 22: max_depth



From the above figure, it is seen that the model has improved but still needs more values to improve the accuracy for the test data.

The next step is to post prune it and find the best ccp_alpha, Cost complexity pruning. Cost complexity pruning provides another option to control the size of a tree. In DecisionTreeClassifier, this pruning technique is parameterized by the cost complexity parameter, ccp_alpha. Greater values of ccp_alpha increase the number of nodes pruned. 

To find the best value of ccp_alpha, it is required to plot the AUC-ROC score vs the alpha value to see where the accuracy is maximum for both training and test datas.
 ![image](https://user-images.githubusercontent.com/33021726/149593157-c4570b45-0a5b-4621-b468-d91ff5fd24dd.png)

Figure 23: ccp_alpha

Output:
 ![image](https://user-images.githubusercontent.com/33021726/149593177-f0db3e06-63fb-4c7b-8b17-b79b6bc3cc50.png)

Figure 23: ccp_alpha to accuracy graph

From the above figure the maximum accuracy the model can reach is when the alpha value is at 0.001. Now combining all the values gives us an output of:
![image](https://user-images.githubusercontent.com/33021726/149593183-8ced6ac0-725a-49e1-9cd6-4633594e1558.png)
 
Figure 24: ccp_alpha evaluation

 ![image](https://user-images.githubusercontent.com/33021726/149593225-4e966c65-c2d5-433d-97c4-220b413635f5.png)

Figure 25

Performance Evaluation:
 ![image](https://user-images.githubusercontent.com/33021726/149593235-46c9ae1e-9a04-49c6-900b-3c1e5f7c660c.png)

Figure 26: Performance Evaluation

Now the accuracy for the test data is greater than the accuracy for the training data, which significantly increases the model’s capability of classifying new data.

Visualizing the tree:
 ![image](https://user-images.githubusercontent.com/33021726/149593244-6cba2853-ad5d-4fff-ac1c-4296fa63d626.png)

Figure 27: Plotting the Decision Tree

Output:
 ![image](https://user-images.githubusercontent.com/33021726/149593255-31fe6989-ef27-4733-bea3-bde7a777046f.png)

Figure 28: Decision Tree

This is what the tree looks like, as it can be seen the maximum depth it has undergone is 10 and the root node in the tree looks like the figure below:
 ![image](https://user-images.githubusercontent.com/33021726/149593270-31af93b4-0fda-4488-ab64-e7a5eeea3a2c.png)

Figure 29: Root Node

The last step is to check the feature importance in the model:
 ![image](https://user-images.githubusercontent.com/33021726/149593283-534268a2-b3e7-46da-b0f9-1c51165dcd99.png)

Figure 30: Plotting Feature Importance


Output:
 ![image](https://user-images.githubusercontent.com/33021726/149593294-5a5f7a5e-aa6b-4d5c-9e21-11d975bab651.png)

Figure 31: Feature Importance

b.	Random Forest Algorithm

Random forest is a flexible, easy-to-use machine learning algorithm that produces, even without hyper-parameter tuning, a great result most of the time. It is also one of the most used algorithms, because of its simplicity and diversity (it can be used for both classification and regression tasks). It is a supervised learning algorithm. The "forest" it builds is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.

The first step for Random Forest is to determine the best possible values of its parameters, i.e. n_estimators (This is the number of trees you want to build before taking the maximum voting or averages of predictions. A higher number of trees give you better performance but makes your code slower.), max_depth, and min_samples_leaf.

To determine the best value of n_estimators we use GridSearchCv to plot the AUC_Score with respect to n_estimators.

![image](https://user-images.githubusercontent.com/33021726/149593308-2dee2304-9a27-4c42-8063-3a0e88ccca62.png)
 
Figure 32: n_estimators
Output:
 ![image](https://user-images.githubusercontent.com/33021726/149593317-4e99f507-9fc6-4a58-bdcc-afb84aefd516.png)

Figure 33: n_estimators to AUC-Score

The next step is to find out the best value for max_depth. That is also done using GridSearchCv in the following way:

![image](https://user-images.githubusercontent.com/33021726/149593331-61ba9f15-fd81-4dc7-bb5b-06ee2a05ea53.png)
 
Figure 34: max_depths

 ![image](https://user-images.githubusercontent.com/33021726/149593339-22ac2365-0ed8-4670-9024-d95a8383bfa0.png)

Figure 35: max_depths to AUC Score
Now that we know the best values for max_depth and n_estimator, we can use the Random Forest Classifier to use those values and predict the target values.
![image](https://user-images.githubusercontent.com/33021726/149593353-8bc3d172-b921-43b1-9569-50b2d28197eb.png)

Figure 36: Random Forest Classifier
 ![image](https://user-images.githubusercontent.com/33021726/149593359-9581efbf-68d0-430c-bc1a-14d9dbe63986.png)

Figure 37

Performance Evaluation:
 ![image](https://user-images.githubusercontent.com/33021726/149593367-fa6fb0db-b1ed-4ed7-9da7-0a708eb8465b.png)

Figure 38: Random Forest Classifier Performance Evaluation





Plotting a Random Tree is computationally costly cause it computes and calculates 100 trees:
 ![image](https://user-images.githubusercontent.com/33021726/149593377-a93bfa8a-ad9d-4a94-825f-f6f98b4224fb.png)

Figure 39: Plotting Random Forest

Output:
The tree is big and takes quite a long time to run, this is some of the root nodes on the tree:
 ![image](https://user-images.githubusercontent.com/33021726/149593381-c522882a-ddc8-4b39-a31a-d1daade0b6e0.png)

Figure 40: Random Forest Tree

 ![image](https://user-images.githubusercontent.com/33021726/149593391-8ad82b99-8fbd-40ce-b31c-f81f87985820.png)

Figure 41: Random Forest Tree sample

The last step is to check the feature importance in the model:
 ![image](https://user-images.githubusercontent.com/33021726/149593399-429000ba-6fbb-4a19-81f7-3490094b4dbf.png)

Figure 42: Plotting Random Forest Feature Importance

Output:
 ![image](https://user-images.githubusercontent.com/33021726/149593415-1c0e741c-010e-4e04-a450-de78f7999af0.png)

Figure 43: Random Forest Feature Importance

c.	SVM Algorithm
In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.


 ![image](https://user-images.githubusercontent.com/33021726/149593425-8094b428-2bbe-4533-9a0a-3c598ede7ec9.png)

Figure 44: svm.SVC

After we fit the model, we are ready to use it to predict the test data.
![image](https://user-images.githubusercontent.com/33021726/149593437-b9f31e7f-36c9-4a1a-9d90-8ea45fda920b.png)
 
Figure 45: SVM accuracy score

Performance Evaluation:
![image](https://user-images.githubusercontent.com/33021726/149593449-87d95506-9010-4d01-8064-bd2ebe8bed9b.png)
 
Figure 46: SVM performance evaluation

9.	Comparison of Models 

To compare the models’ performance, a data frame is created. The columns of this data frame are the lists created to store the results of the model.
![image](https://user-images.githubusercontent.com/33021726/149593462-9241ac69-674d-44b5-828b-f884e56d58fc.png)
 
Figure 47: Comparisons of Model

The models are evaluated, and the considered metric is accuracy. From the above figures, it is shown that Random Forest gives better performance. The last step is to save and load the model and test out the model on some sample data.

 ![image](https://user-images.githubusercontent.com/33021726/149593479-412613ca-4db6-437b-9217-d30e9c0eb7f8.png)

Figure 48: Loading and testing Model


Predicting the data:

 ![image](https://user-images.githubusercontent.com/33021726/149593485-50f62c9c-9595-4733-b672-e12fa114984f.png)
 
Figure 49: Predicting Result

This test was conducted on an 80-20 split, further splitting testing was done. The comparison of the models when the data is split on an 80-20, 60-40, and 50-50 are:

 ![image](https://user-images.githubusercontent.com/33021726/149593491-5b60b1af-dd65-4600-9891-aa74e6b529b2.png)

Figure 50: 80-20 split data performance evaluation	 
 ![image](https://user-images.githubusercontent.com/33021726/149593501-2cf8032f-0bb6-41fa-a704-84bcb852af84.png)

Figure 51: 60-40 split data performance evaluation
 ![image](https://user-images.githubusercontent.com/33021726/149593510-d4b796d8-5395-4626-988c-549408d83780.png)

Figure 52: 50-50 split data performance evaluation	
Table1: Comparison of Models

In all cases, the Random Forest algorithm performs much better than the rest.


10.	Conclusion

This project aims to enhance detection methods to detect phishing websites using machine learning technology. Detection accuracy of 97.0% was achieved using the Random Forest algorithm with the lowest false positive rate. The result shows that classifiers give better performance when more instances and more attributes was used. 

The problem of phishing cannot be eradicated, nonetheless can be reduced by combating it in two ways, improving targeted anti-phishing procedures and techniques and informing the public on how fraudulent phishing websites can be detected and identified. To combat the ever evolving and complexity of phishing attacks and tactics, ML anti-phishing techniques are essential.

The outcome of this project reveals that the proposed method presents superior results rather than the existing deep learning methods. It has achieved better accuracy and F1—score with a limited amount of time. The future direction of this project is to develop an unsupervised deep learning method to generate insight from a URL.


The algorithms used on this dataset have never been used on it. The few other projects done are on different datasets with not many attributes, I even found out missing attributes from other datasets that were very important to the decision making. It’s a new data with new attributes, I believe my contributions are using this data and using the classifier algorithms and post and pre pruning the data to avoid overfitting was one of the challenges. Other projects and literature reviews I have read have not applied this classification problems to this kind of data, with this many attributes, they have not used this new and updated dataset and have not applied Pre-Pruning and Post Pruning. So that was a first and it really made the data easy to apply my models onto it. 


Working on this project was very knowledgeable and worth the effort. In the future, this project could be used as a browser extension or an application with a user interface that users can use to detect phishing websites more easily.


11.	     References

[1] (PDF) Phishing Website Detection using Machine Learning Algorithms (researchgate.net)

[2] Phishing URL Detection with ML. Phishing is a form of fraud in which… | by Ebubekir Büber | Towards Data Science

[3] Phishing-Website-Detection-by-Machine-Learning-Techniques

[4] UCI Machine Learning Repository: Phishing Websites Data Set

[5] Detecting phishing websites using machine learning technique (plos.org)

[6] Phishing - Wikipedia

[7] Phishing Definition (investopedia.com)



#OBJECTIVE
Given information about a user — how many pages they’ve visited, whether they’re shopping on a weekend, what web browser they’re using, etc. — 
our classifier will predict whether or not the user will make a purchase. Our classifier won’t be perfectly accurate — but it will be better than guessing randomly. 
To train our classifier, we’ll use data from a shopping website with ~12,000 users sessions.

#RUNNING
When we run our program, we pass the location of a CSV file containing data about website shoppers as a command line argument. 
Our program takes that data, reformats it into the correct data type, splits it into training and testing data, uses that data to 
train a k-nearest neighbors classification model,  outputs the sensitivity (true positive rate) and specificity (true negative rate) 
of our trained model on our testing data. 

#MY RESPONSIBILITIES
load_data() - The function loading in the data from the CSV and reformatting it before sending it to our model
train_model() - The k-nearest neighbor model function
Evaluate() - The function measuring our true positive/true negative stats on the testing data

The main function was provided by Brian Yu over at harvard. 

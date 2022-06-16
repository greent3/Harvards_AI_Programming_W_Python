import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    """
    with open(filename, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        evidence_list = []
        labels_list = []

        # loop through each row in CSV file
        for row in csv_reader:
            inner_list = []

            # loop through each cell in current row and cast to appropriate data type before appending to evidence list
            for i in range(len(row) - 1):
                if i in [0, 2, 4, 11, 12, 13, 14]:
                    inner_list.append(int(row[i]))
                elif i in [1, 3, 5, 6, 7, 8, 9]:
                    inner_list.append(float(row[i]))
                elif i == 10:
                    inner_list.append(month_calculator(row[i]))
                elif i == 15:
                    if row[i] == 'Returning_Visitor':
                        inner_list.append(1)
                    else:
                        inner_list.append(0)
                else:
                    if row[i] == 'FALSE':
                        inner_list.append(0)
                    else:
                        inner_list.append(1)
            
            # row[17] (column 16) contains our label 
            if row[17] == 'FALSE':
                labels_list.append(0)
            else:
                labels_list.append(1)
            evidence_list.append(inner_list)

        new_tuple = (evidence_list, labels_list)
        return new_tuple


def month_calculator(string):
    """
    Helper function for load_data
    to change a month represented
    as a string, to an integer
    """
    month_dict = {
        'Jan': 0,
        'Feb': 1,
        'Mar': 2,
        'Apr': 3,
        'May': 4,
        'June': 5,
        'Jul': 6,
        'Aug': 7,
        'Sep': 8,
        'Oct': 9,
        'Nov': 10,
        'Dec': 11
    }
    return month_dict[string]


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(evidence, labels)
    return knn


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).
    """
    actual_purchases = 0
    actual_refusals = 0
    correct_purchases = 0
    correct_refusals = 0

    # loop through labels and predictions, tallying up correct & incorrect predictions
    for i in range(len(labels)):
        if labels[i] == 1:
            actual_purchases += 1
            if predictions[i] == 1:
                correct_purchases += 1
        else:
            actual_refusals += 1
            if predictions[i] == 0:
                correct_refusals += 1

    sensitivity = float(correct_purchases / actual_purchases) # true positive rate
    specificity = float(correct_refusals / actual_refusals) # true negative rate
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()

import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP Conj NP VP | NP VP NP Conj NP VP Adv | NP VP Conj VP NP | NP VP PP Conj VP PP | NP VP PP
VP -> V | V NP | V PP | Adv V | V Adv
NP -> N | Det N | P N | Adj N | Det N P N | Det Adj N | Det N P Det N
PP -> P NP
Adj -> Adj Adj | Adj Adj Adj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    # Make sentence lowercase
    sentence = sentence.lower()
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
     'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    word_arr = nltk.word_tokenize(sentence)

    # Remove any word without an 'a-z' character from our list 
    list_of_words = []
    for word in word_arr:
        contains_one = 0
        for char in word:
            if char in alphabet:
                contains_one = 1
        if contains_one == 1:
            list_of_words.append(word)
    return list_of_words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_list = []
    for s in tree.subtrees(lambda t: t.label() == 'NP'):
        if no_np_subs(s) is True:
            np_list.append(s)
    return np_list


def no_np_subs(subtree):
    """
    Recursive helper function for np_chunk. Returns true if
    no NP chunks exist further down the tree. Returns false
    otherwise.
    """
    # Recurse through our given tree's subtrees, checking for Noun Phrases
    for index in subtree.subtrees(lambda t: t != subtree):
        if index.label() == 'NP':
            return False
        else:
            return no_np_subs(index)
    return True


if __name__ == "__main__":
    main()

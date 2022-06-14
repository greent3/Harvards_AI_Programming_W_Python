OBJECTIVE
Our objective is to take a sentence full of non-terminals, or parts of a sentence that can be broken down into a smaller part of the sentence, 
and use the NLTK library to parse them down into terminals.

NLTK is really cool and actually makes it really easy to parse a sentence based on your grammar rules, 
but we want to go the extra mile and separate our noun phrases from our parsed sentence.

This is important in ML because Noun phrases are often what our algorithms use to understand what a sentence is about, and then make predictions based on that meaning.

RUNNING
So if we type in something like “Holmes satin the red armchair”
We’ll not only get a printout of our parsed sentence, but also our lowest noun_phrase chunks in our tree of parsed terminals and non-terminals.

MY RESPONSIBILITIES
I completed this project as a part of Harvard’s AI Programming with Python course. 
The main() function and terminal definitions were provided by Brian Yu over at Harvard, 
while I wrote preprocess(),  np_chunk(), no_np_subs(), and defined the nonterminal logic for NLTK.

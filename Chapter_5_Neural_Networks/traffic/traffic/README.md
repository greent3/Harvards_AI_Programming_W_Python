Convolutional layers:
Doubling the amount of convolutional layers significantly increased the accuracy of the program, and decreased the loss,
Meanwhile, tripling (total of 3 convolutional layers) had a very minimal effect.

Convolutional filters:
An increased amount of filters (from 32 each layer to 132 each layer) offered very little in terms of accuracy and
it significantly decreased the speed of the program. (24s per epoch vs 6s per epoc)

Dropout:
I noticed decreasing the dropout percentage significantly increased the accuracy and decreased loss, though I am sure this is from overfitting

Max Pool Size:
Increasing the size of the 2D pooling from 2x2 to 4x4 decreased accuracy and increased loss, but shortened the programs runtime by almost half!

In summary, improvements to the program's accuracy comes at the cost of the programs speed.
Some improvements to the algorithm were worth the time-cost. (Ex: adding a 2nd convolutional layer increased 
my program's accuracy from .05 to 0.95, well worth the time-cost!)
The time-cost appears to be exponentially increasing after reaching a certain upper-threshold of accuracy.
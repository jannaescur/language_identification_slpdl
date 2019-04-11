# n-grams approach

After doing some research on language identification deep learning approaches, we found that many of them use words 
or ngrams level inputs, instead of characters. This can work because even RNNs (LSTM in our case) can learn about sequences 
(which would be the text) the fact of having specific words per language or n-grams can define more precisely the behaviour of
each language.<br>
We first tried a word level approach, which did not predict good results, so we decided to keep the character approach but
using n-grams. To start with n-grams, first we decided to 
use n=3 (trigrams) at a character level. It was the most reasonable number to start with, as it was closer to the character baseline approach and
also because of the restrictiveness of the computational resources (with n=5 or n=7 the number of possible tokens was
considerably increased).<br>
Altough we know that n-grams methods can take advantadge of the prior probability of each n-gram (in the training set), to start
with we directly changed the input tokens from one character to trigrams. An n-gram can be defined as a sequence of contiguous 
words in a text, so in our case we built a sequence of 3 contiguous characters, and indexes were assigned to every trigram
(our tokens).<br>
We did not have to change the structure of the network, only the way of how the dictionary was created. Even so, we used
some of the changes done in our first approach "hyperparameters optimization", as the Dropout (with p=0.5), the hidden size reduced at 64,
the learning rate decay (every 4 epochs reduced by 0.5)... and instead of adding regularization at the loss, 
we decided to try weight decay in the optimizer (Adam), which has the same effect, with a decay of 4e-5:<br>
*optimizer = torch.optim.Adam(model.parameters(), weight_decay=4e-5)<br>*

Using this new and more interesting approach we reached an accuracy of 91.1%, which is very close to the baseline.<br> (*Notebook:
trigrams-lstm.ipynb*)<br><br>

**Adding probabilities:**<br>

With the n-grams approach happened the same as the baseline in the terms of overfitting: we still got a very high train 
accuracy (99%), while the validation accuracy did not increase that much. Thinking about how this could be improved
with n-grams (not focusing on avoiding overfitting with neural network tunning: this is done in approach "hyperparameter optimization)
we decided to take advantage of probabilities. The first idea was to remove (mark as unknown) the most probable n-grams per language,
so we would not have redundancy and the network could learn better from the fed data. Do to the computational restrictions of
the Kaggle kernel execution, it was pretty hard to make it work, so we finally decided to remove the n-grams that appear more
times in the training set without taking into acount its labels (to which language they belong).<br>

With this approach we got an accuracy of 92.3%, a little bit higher than using n-grams without the probability step.<br> (*Notebook: trigrams-with-probabilities.ipynb*)

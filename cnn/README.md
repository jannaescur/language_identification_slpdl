# CNN for language identification
##  Motivation: 
Modify the baseline architecture so that we can use a CNN for the language detection in order to reduce the inference time while keeping similar results as LSTM.

## Schema:
Instead of applying a recursive network we will format the embeddings in a 1d array of shape: [b_size, embedding_size*#words]. This way we are able to apply 1d convolutions of the text with kernel_size= 3*embedding size (else we would be slicing our filters inside the same word embedding)

The schema of the network are 3 convolution layers with 512 filters of different kernel_sizes that when combined provide different resolution context features (3 words, 4 words, 5 words) which is forwarded to the final classifier (fully connected layer).

 <p align="left">
  <img src="captura.png"/>
</p>

Even though the idea looked promising we believe it would be more useful for semantic problems where we could even load pretrained embeddings such as GloVe thant using it for language identification where the characters may be more discriminative than words.

## Conclusions:
After having a lot of problems to make the net work we achieved to make it learn even though the performance is far far away from the LSTM character level baseline.

Max(Training accuracy) = 76.73242490197241
Min(Training loss) ) = 0.7


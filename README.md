# This is a LSTM RNN that is able to generate text based on multiple txt files provided by user.

In order to use the *jupyter notebook file*, you will need to change the file paths to files with texts in it. When using *.py* file, you will be prompted to select the folder with texts.

## What is the idea?
The general idea is to learn the dependencies between words and the conditional probabilities of words in sequences so that we can in turn generate wholly new and original sequences of words. We recommend using books from project Gutenberg (https://www.gutenberg.org). It gives access to free books that are no longer protected by copyrights.

## How it works?
We split the book text up into subsequences with a fixed length of 11 words, an arbitrary length. We could just as easily split the data up by sentences and pad the shorter sequences and truncate the longer ones.

Each training pattern of the network is comprised of 11 time steps of one word (X) followed by one word output (y). When creating these sequences, we slide this window along the whole book one word at a time, allowing each word a chance to be learned from the 11 words that preceded it (except the first 11 words of course).

For example, if the sequence length is 5 (for simplicity) then the first two training patterns would be as follows:
The next day we went -> to
next day we went to -> breakfast

## Preparing data and defining the model
To prepare the data, we use word embeddings, so that it is suitable for use with Keras.

We can now define our LSTM model. Here we define a single hidden LSTM layer with 256 memory units. The output layer is a Dense layer using the softmax activation function to output a probability prediction between 0 and 1.

We are not interested in the most accurate (classification accuracy) model of the training dataset. This would be a model that predicts each word in the training dataset perfectly. Instead we are interested in a generalization of the dataset that minimizes the chosen loss function. We are seeking a balance between generalization and overfitting but short of memorization.

## Generating Text with LSTM network
Firstly, we load the data and define the network in exactly the same way, except the network weights are loaded from a checkpoint file and the network does not need to be trained.

The simplest way to use the Keras LSTM model to make predictions is to first start off with a seed sequence as input, generate the next word then update the seed sequence to add the generated word on the end and trim off the first word. This process is repeated for as long as we want to predict new words (e.g. a sequence of 50 characters in length).

We can pick a random input pattern as our seed sequence, then print generated words as we generate them.
The generated text with the random seed (based on Mobby Dick and Alice in Wonderland) was:

*the leeward beach of the rigging with the riveted canal i uttered a row arrah a vengeance i joyously hospitals and sheered away through oceans coffin and wherefore have a grudge with indispensable the obsequious hint of fifty fireengines that is drawn out again with those bows and welded to*

We can see that generally there are fewer spelling mistakes and the text looks more realistic, but is still quite nonsensical.

**Requirements:**
- python3
- packages from requirements.txt

Tensorflow-gpu installation guide: https://youtu.be/qrkEYf-YDyI

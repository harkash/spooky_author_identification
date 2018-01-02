# spooky_author_identification
This is a submission to the Kaggle challenge - Spooky Author Identification.

In this challenge, the task is to identify authors from their text. The data contains three authors - Edgar Allan Poe, Mary Shelley, and HP Lovecraft.

The basic approach is to first create Glove embeddings of the text and then apply various recurrent neural network architectures to see how they perform. 
The approaches are - 1d Convolution, LSTM, bi-directional LSTM and GRU.

The multi-class log loss for the models are (obtained after submitting on Kaggle) - 
* 1D CNN - 2.51
* LSTM - 0.58805
* Bi-directional LSTM - 0.60228
* GRU - (not tried yet)
* CNN + LSTM - 0.725


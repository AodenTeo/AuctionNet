# ArtNet: Auction Price Prediction

The program consists of three main files described below: 

+ **process.py**: 

  + Prepares the economic data to add to the training set, as well as normalizes all prices to be in *real USD*. It writes all of this updated data to clean_art.csv. None of these functions are called anywhere else. 

+ **model.py**: (*alert:* MISLEADING NAME - should be called stringproc.py)

  + Contains helper functions for string processing, and vocabulary building (building a dictionary of words that appear in the dataset). The important functions are: 

    + **tokenize**: Takes in a Python array of strings, and splits each string into an array with each word separate, and each word taken to lower case. 

      + *Example*: 

      + ```python
        tokenize(["Hello, world", "Here's another string"])
        
        ### @RETURNS #### 
        [["hello", ",", "world"], ["here", "is", "another", "string"]]
        ```

    + **create_vocab**: creates a vocabulary object out of a python array of strings.

      + *Example:*

      + ```python
        create_vocab(["Hello", "world"])
        
        ### @RETURNS: a vocab object that can be accessed as follows #### 
        vocab["hello"] ### returns 0 (because hello is the first word in the vocab)
        vocab["world"] ### returns 1 (because world is the second word in the vocab)
        ```

    + **create_vocab**: creates a vocabulary object out of a column of a CSV file 

      + *Example*: 

      + ```python
        create_vocab("csv_file_path.csv", "Artist")
        
        ### @RETURNS: a vocab object that can be accessed as follows #### 
        vocab["hello"] ### returns 0 (because hello is the first word in the vocab generated from the Artist column)
        vocab["world"] ### returns 1 (because world is the second word in the vocab generated from the Artist column)
        ```

    + **text_to_tensor**: takes in an array of strings, and a vocabulary, and replaces each string with a tensor where each word in the string is replaced by its integer index in the vocabulary. 

      + *Example:*

      + ```python
        text_to_tensor(["Hello my name is Bob", "I am flying", "To be or not to be"])
        
        #### @RETURNS ####
        tensor([[1.0, 5.0, 3.0, ...], ...]) # where "Hello" is the first word in the vocabulary, "my" is the fifth word in the vocabulary,... 
        ```

+ **scratch.py**: (*alert*: MISLEADING NAME - should be named dataload.py)

  + Creates a PyTorch dataset object (which can be used to represent datasets later in code) called ArtDataset, which loads in data from the CSV file, does some final preprocessing (replaces each string with its distance from the word museum) and sets some attributes of the dataset object, listed below
    + dataset.vocab_artist: vocab object representing the vocabulary used in the artist column
    + dataset.vocab_title: vocab object representing the vocabulary used in the title column
    + dataset.artist_col_numerical: Torch tensor representing the average distance of each word from the word museum
    + dataset.artist_string: Array of strings in the artist column (for visualising the model's predictions later) (NOT USED FOR TRAINING, USED FOR DATA VISUALIZATION)
    + dataset.artist: Output of text_to_tensor on the artist column (NOT USED ANYMORE)
    +  dataset.title_numerical: Torch tensor representing the average distance of each word from the word museum
    + dataset.title_string: Array of strings in the title column (for visualising the model's predictions later) (NOT USED FOR TRAINING, USED FOR DATA VISUALIZATION) 
    + dataset.title: Output of text_to_tensor on the title column (NOT USED ANYMORE)
    + dataset.numerics_mean: Vector with the average of all the columns of economic data
    + dataset.numerics_std: Standard deviation of all the columns of economic data
    + dataset.numerics: Tensor with all the numerical economic data that we use to train the model
    + dataset.x: Tensor with all the input features we will feed to the model
    + self.price_median: Medians of all the prices
    + self.price_std: Standard deviation of all the prices
    + self.price: Tensor with all the prices is in it (labels for the model to learn)
    + self.artist_vocab_len: Length of the artist vocabulary
    + self.title_vocab_len: Length of the title vocabulary

+ **train.ipynb**: 

  + Notebook which loads the dataset, trains the model  

+ **model_weights.txt**:

  + File to which our most successful model weights are saved

  
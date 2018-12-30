# Sequence Model for generating shakespeare like text
* We take input of 4 words and predict 5th word of the sentence using simple word level 
    rnn
* The dataset consists of text files from play king henry 4 and is stored in directory data/
* The processed data directory will contain .npy files created after running
     
    ```
   python code/preprocess.py
    ```
 *  To train and generate text use 
   
   ```
  python code/nn.py
    ```
   
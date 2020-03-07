---
layout: post
title:  Introduction To The Fast Fourier Transform In Python
date:   2017-11-01
status: published
---

Last week, I was bored. 


This article is an attempt to document everything I learnt while working with `char-rnn` and how I was able to generate yoruba names with it.


The rest of the article will be organised as follows:

* What is `char-rnn`? 
* Yoruba Name Generation
    * Inspiration
    * Getting the data
    * Choosing the approach
    * Result Benchmark, Challenges and Improvements
    * Some of the generated names
* Resources  

### What is `Char-RNN`
`Char-rnn` short for `Character Recurrent Neural Network` are multi-layer neural networks for training textual models. They are essentially made up of RNN, LSTM and GRU.
According to the author, [Andrej Karperthy](),

>>> The (`char-rnn`) model takes one text file as input and trains a Recurrent Neural Network that learns to predict the next character in a sequence. The RNN can then be used to generate text 
character by character that will look like the original training data. 

Given the above, a `char-rnn` model can be thought of classification model in that it's output is a probability distribution of character classes.
The model is fed the characters one at a time and an output is generated at each ending character.

Since this is a classification task, categorical cross entropy loss is used to train the model. This loss is mathematically given as 

$$ \sum_{i=1}^{n}y_{i}log(Y_{i})\label{c}\tag{1} $$

where $$n$$ represents number of samples

where $$y_{i}$$ represents encoded target value(character in this instance)

where $$Y_{i}$$ represents vector of probabilities over the samples

The target characters(values) are derived by one hot encoding them as one-hot vectors. The position of the `1` represents the position of the
generated character. The generated characters are created by iteratively minimizing the loss in `equation 1` above.


### The Data
The Yoruba names database contains 6842 of which 6280 were published. I would assume the 562 unpublished names were as a still undergoing scrutiny for errors or were not right for the 
platform.

Each name had the following structure:

```json
[
  {
    "id": 1,
    "pronunciation": "",
    "ipaNotation": "",
    "variants": "Àrẹ",
    "syllables": "à-à-rẹ",
    "meaning": "The one who is the chief.",
    "extendedMeaning": null,
    "morphology": "ààrẹ",
    "geoLocation": [
      {
        "place": "OTHERS",
        "region": "OTHERS"
      }
    ],
    "famousPeople": "Bọ́lá Àrẹ, gospel musician",
    "inOtherLanguages": null,
    "media": "https://en.wikipedia.org/wiki/Bola_Are",
    "tonalMark": null,
    "tags": null,
    "submittedBy": "Not Available",
    "etymology": [
      {
        "part": "ààrẹ",
        "meaning": "chief"
      }
    ],
    "state": "PUBLISHED",
    "createdAt": [
      2016,
      1,
      7,
      21,
      18,
      1,
      843000000
    ],
    "updatedAt": [
      2016,
      11,
      5,
      11,
      10,
      55,
      814000000
    ],
    "name": "Ààrẹ"
  }
]
```

Getting this data was easy. All it took was a saved `wget` request to their [API](yorubaname.com/swagger-ui.html).

```bash

wget 'http://yorubaname.com/v1/names?count=6842' -O yorubanames.json
```

Since the task is to generate names, after getting the data, every other field in the dataset asides the `name` field was dropped. This is
majorly because ``char-rnn`` cares less about feature samples and more about the characters that make up the needed label column. Another reason 
would be because, I personally, haven't found a way to use tabular data with multiple features on ``char-rnn``.

The dataset had 6284 entries which is somewhat very `small` and has been made available [here](https://github.com/Olamyy/created_datasets).

### Choosing the approach

There were a number of different approaches I could have taken while working on this.
Some of these approaches are:

1. Using the Pytorch implementation of ``char-rnn`` [here](https://github.com/karpathy/char-rnn].com/karpathy/char-rnn)
2. Using this theano/lassagne [implementation](https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py)
3. Using [Hendrik Weideman](https://github.com/hjweide/lasagne-char-rnn)'s lassagne implementation
4. Using the [textgenrnn](https://github.com/minimaxir/textgenrnn) python package
5. Writing my own custom implementation as a way to learn.

I started by writing my own custom [implementation]() of ``char-rnn`` but the performance was a bit poor. I ended using `textgenrnn` which is
a Keras implementation of the concept.

An added advantage of using textgenrnn is easy setup with fewer lines of code. A fully working instance of char-rnn can be set up using
the library in as little as 8 lines of code.

For my Yoruba names use case, I was able to setup a fully running char-rnn model on the data using the code below:

```python
from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('yorubanames.csv', num_epochs=5)  #yorubanames.csv contains the
#  6842 names available on the yoruba names platform.

textgen.generate(5)

```

#### Result Benchmark, Challenges and Improvements.

Using the default textgenrnn config, the model seems to perform quite well, generating names, 
interesting enough to be taken as original yoruba names. However, most(83%) of the names were monosyllabic which would suggest they were more of actions(verbs) than names.


To improve this, I tried to get the optimal hyperparameters for the model. Using ``sklearn``'s GridSearch algorithm, I was able to conclude on the following parameters as the 
best performing.

```python
cfg = {'line_delimited': False,
       'num_epochs': 12,
        'gen_epochs': 4,
        'train_size': 0.8,
        'validation': True,
        'is_csv': True
        }
```

Training the data with this config very much improved the generated names. The names were now made up of either 4 or 3 syllables which was much more acceptable.


While this is nowhere near perfection, some of the challenges I faced while doing this include:

1. Understanding how the model chooses yoruba signs (amin) for each name.
2. For some weird reason, the model seems to perform way better(generate better names) when I export the dataset with its numerical index than when there's no index. I've tried investigating this
but haven't been able to come up with a good explanation as to why this happens.
3. The library does not provide access to all Keras Model APIs so there was really no empirical way to measure its performance asides manual calculations.

Some possible improvements on my initial work would be :

1. Look for a way to include all features of the yoruba names dataset in the training. This would mean, looking for a way to include tabular multi feature datasets in `char-rnn`.
2. Improve the textgenrnn package to make it a bit more pythonic.
3. Generate meanings and along with each name. This would help provide more context as to what the algorithm perceives the meaning of the name to be.


## Cross section of generated names

The first 50 generated names are shown below

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ifádájù</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Àyínàjà</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ḿrìnní</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aríyéyé</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Àjẹ̀yà</td>
    </tr>
    <tr>
      <th>4</th>
      <td>láyèsì</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adeguntà</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Olúwanmíĺlá</td>
    </tr>
    <tr>
      <th>7</th>
      <td>antà</td>
    </tr>
    <tr>
      <th>8</th>
      <td>̀tẹ̀t́ládé</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Adéyìntà</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Àdẹ̀yàn</td>
    </tr>
    <tr>
      <th>11</th>
      <td>oshiníyè</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Olúwat̀mí</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Olúwátóyè</td>
    </tr>
    <tr>
      <th>14</th>
      <td>láníyì</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Akínfádù</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ògbélá</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Adégbé</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Olúwàtàn</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Àyìbádé</td>
    </tr>
    <tr>
      <th>20</th>
      <td>̀jẹ́yì</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Ifádéyé</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ifadòyíní</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Adébíyì</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Fátóyè</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Adémilé</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Olúwákóyé</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ĺruníĺlá</td>
    </tr>
    <tr>
      <th>28</th>
      <td>ḿdùnmí</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Fádáyè</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Àjòyí</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Àyínlẹ́yé</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Ifádéyé</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Olúwabíróyè</td>
    </tr>
    <tr>
      <th>34</th>
      <td>láníyè</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Akínráyì</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Olúwáwó</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Ìdẹ̀jú</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Olúwaranísan</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Ìbóyè</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ólá</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Adégbé</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Àbẹ́lá</td>
    </tr>
    <tr>
      <th>43</th>
      <td>oloolúwadá</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Olúwáshinílá</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Akínbẹ́mi</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Báyé</td>
    </tr>
    <tr>
      <th>47</th>
      <td>oluntì</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Olúwáníyì</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Àyínká</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Máyìn</td>
    </tr>
  </tbody>
</table>




## Resources
1. [https://github.com/karpathy/char-rnn](https://github.com/karpathy/char-rnn)
1. [https://github.com/minimaxir/textgenrnn](https://github.com/minimaxir/textgenrnn)
1. [https://github.com/ekzhang/char-rnn-keras/blob/master/Visualization.ipynb](https://github.com/ekzhang/char-rnn-keras/blob/master/Visualization.ipynb)
1. [https://stackoverflow.com/questions/41484580/gridsearch-with-keras-neural-networks](https://stackoverflow.com/questions/41484580/gridsearch-with-keras-neural-networks)
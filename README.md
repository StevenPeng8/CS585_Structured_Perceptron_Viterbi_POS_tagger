# Structured Perceptron and Viterbi Based Part of Speech Tagger

### A project for CS585 - [Introduction to Natural Language Processing](http://people.cs.umass.edu/~brenocon/inlp2016/)

### Assignment Description ([Viterbi](http://people.cs.umass.edu/~brenocon/inlp2016/hw4/hw4a.pdf), [Perceptron](http://people.cs.umass.edu/~brenocon/inlp2016/hw4/hw4b.pdf))

### [Starter code](http://people.cs.umass.edu/~brenocon/inlp2016/hw4/structperc.py)

### Data ([training](http://people.cs.umass.edu/~brenocon/inlp2016/hw4/oct27.train), [development](http://people.cs.umass.edu/~brenocon/inlp2016/hw4/oct27.dev))

### Instructor: [Brendan T. O'Connor](http://brenocon.com/)

## Description

Trains a Structured Perceptron Linear Classifier to tag parts of speech using the Viterbi algorithm for decoding. The assignment code has been cleaned up and streamlined to facilitate reading and usage. This means the complete solution to the assignment is not here, just what I deemed the most relevant part for sharing.

## Instructor Implementations

### baseline.py

All

### vit.py

* `dict_argmax`
* `goodness_score`
* `exhaustive`
* `randomized_test`

### structperc.py

* `dict_subtract`
* `dict_argmax`
* `dict_dotprod`
* `read_tagging_file`
* `do_evaluation`
* `fancy_eval`
* `show_predictions`
* `greedy_decode`

## Modifications to Instructor Implementations

### structperc.py

* `local_emission_features`: Added suffix features
* `train`: Implemented inner loop, core of the training algorithm. Instructor code just a skeleton.

## Implementations I provided

### vit.py

* `viterbi`

### structperc.py

* `get_averaged_weights`
* `predict_seq`
* `features_for_seq`-
* `calc_factor_scores`-

## Demo

To train a tagger with 10 iterations of structured perceptron, using viterbi:

`python structperc.py`

`baseline.py` checks the accuracy of assuming every word has the same tag. To check this baseline:

`python baseline.py`

## Usage

```
# Import
from structperc import train

# Reads tagging files in the format of oct27.train and oct27.dev
import read_tagging_file

# Train with averaging on the oct27.train data, evaluating with oct27.dev data
train(read_tagging_file('oct27.train'), do_averaging=True, devdata=read_tagging_file('oct27.dev'))
```

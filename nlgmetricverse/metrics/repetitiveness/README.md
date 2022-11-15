# Metric card for Repetitiveness

# Metric description
The [repetition problem](https://github.com/fuzihaofzh/repetition-problem-nlg) has been observed in nearly all text generation models. This problem is, unfortunately, caused by the traits of our language itself. There exists too many words predicting the same word as the subsequent word with high probability. Consequently, it is easy to go back to that word and form repetitions.

The Repetitiveness metric evaluates how many n-grams are repeated on average in the hypothesis sentences, the result is normalized by the length of the sentence.
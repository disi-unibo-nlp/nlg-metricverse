_DESCRIPTION = """\
Perplexity is a common metric for directly assessing generation models by calculating the probability that they assign to
sequences in the test data:
PP(p, x) = \prod_{i=1}^{n}\left(\frac{1}{p(x_{i})}\right)^{\frac{1}{n}}
where $p$ is a model assigning probabilities to elements and $x$ is a sequence of length $n$.
When averaging perplexity values obtained from all the sequences in a text corpus, one should again use the geometric mean:
$mean-PP(p, X) = \exp\left(\frac{1}{m}\sum_{x\in X}\log \textbf{PP}(p, \textbf{x})\right)$
for a set of $m$ examples $X$.
Language modeling---or more specifically, history-based language modeling (as opposed to full sentence models)---is the task
of predicting the next word in a text given the previous words. For instance, we consider the history "Mary likes her coffee
with milk and". Since there's no "right" answer, we'll let our learned model propose a probability distribution over all
possible next words. We say that this model is good if it assigns high probability to "sugar" and low probability to "socks."
Perplexity just measures the cross entropy between the empirical distribution (the distribution of things that actually appear)
and the predicted distribution (what your model likes) and then divides by the number of words and exponentiates after throwing
out unseen words. Perplexity is the inverse of probability and, with some assumptions
(http://www.cs.cmu.edu/~roni/11761/PreviousYearsHandouts/gauntlet.pdf), can be seen as an approximation of the cross-entropy
between the model's predictions and the true underlying sequence probabilities.

BOUNDS
[1, +inf[, where 1 is best.

DIMENSIONS ENCODED
The guiding idea behind perplexity is that a good model will assign high probability to the sequences in the test data.
This is an intuitive, expedient intrinsic evaluation, and it matches well with the objective for models trained with a
cross-entropy or logistic objective.

WEAKNESSES
- Perplexity is heavily dependent on the nature of the underlying vocabulary in the following sense: one can artificially
  lower one's perplexity by having a lot of UNK tokens in the training and test sets. Consider the extreme case in which
  everything is mapped to UNK and perplexity is thus perfect on any test set. The more worrisome thing is that any amount of
  UNK usage side-steps the pervasive challenge of dealing with infrequent words.
- As Hal Daumé discusses in this post (https://nlpers.blogspot.com/2014/05/perplexity-versus-error-rate-for.html), the
  perplexity metric imposes an artificial constraint that one's model outputs are probabilistic.
"""
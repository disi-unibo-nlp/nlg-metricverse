_DESCRIPTION = """\
The Pearson correlation coefficient $\rho$ between two vectors $y$ and $\widehat{y}$ of dimension $N$ is:
$$
pearsonr(y, \widehat{y}) = 
\frac{
  \sum_{i}^{N} (y_{i} - \mu_{y}) \cdot (\widehat{y}_{i} - \mu_{\widehat{y}})
}{
  \sum_{i}^{N} (y_{i} - \mu_{y})^{2} \cdot (\widehat{y}_{i} - \mu_{\widehat{y}})^{2}
}
$$
where $\mu_{y}$ is the mean of $y$ and $\mu_{\widehat{y}}$ is the mean of $\widehat{y}$.
For comparing gold values $y$ and predicted values $\widehat{y}$, Pearson correlation is equivalent to a linear regression using
$\widehat{y}$ and a bias term to predict $y$ (https://lindeloev.github.io/tests-as-linear/).
It is also closely related to R-squared.

Pearson Correlation [15] measures the linear correlation between two sets of data. Spearman
Correlation [73] assesses the monotonic relationships between two variables. Kendall’s Tau [27]
measures the ordinal association between two measured quantities. BartScore.

BOUNDS
[-1, 1], where -1 is a complete negative linear correlation, +1 is a complete positive linear correlation, and 0 is no linear
correlation at all.

WEAKNESSES
Pearson correlations are highly sensitive to the magnitude of the differences between the gold and predicted values. As a result,
they are also very sensitive to outliers. 
"""


_DESCRIPTION = """\
The Spearman rank correlation coefficient between between two vectors $y$ and $\widehat{y}$ of dimension $N$ is the Pearson
coefficient with all of the data mapped to their ranks.

BOUNDS
[-1, 1], where -1 is a complete negative linear correlation, +1 is a complete positive linear correlation, and 0 is no linear
correlation at all.

WEAKNESSES
Unlike Pearson, Spearman is not sensitive to the magnitude of the differences. In fact, it's invariant under all monotonic
rescaling, since the values are converted to ranks. This also makes it less sensitive to outliers than Pearson.
Of course, these strengths become weaknesses in domains where the raw differences do matter. That said, in most NLU contexts,
Spearman will be a good conservative choice for system assessment.
"""

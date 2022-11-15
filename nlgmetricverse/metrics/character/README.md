# Metric Card for CharacTER

## Metric Description
**CharacTER: Translation Edit Rate on Character Level**

CharacTer is a novel character level metric inspired by the commonly applied translation edit rate (Ter). It is defined as the minimum number of character edits required to adjust a hypothesis, until it completely matches the reference, normalized by the length of the hypothesis sentence. CharacTer calculates the character level edit distance while performing the shift edit on word level. Unlike the strict matching criterion in Ter, a hypothesis word is considered to match a reference word and could be shifted, if the edit distance between them is below a threshold value. The Levenshtein distance between the reference and the shifted hypothesis sequence is computed on the character level. In addition, the lengths of hypothesis sequences instead of reference sequences are used for normalizing the edit distance, which effectively counters the issue that shorter translations normally achieve lower Ter.

## Citation(s)
```bibtex
@inproceedings{DBLP:conf/wmt/WangPRN16,
  author    = {Weiyue Wang and
               Jan{-}Thorsten Peter and
               Hendrik Rosendahl and
               Hermann Ney},
  title     = {CharacTer: Translation Edit Rate on Character Level},
  booktitle = {{WMT}},
  pages     = {505--510},
  publisher = {The Association for Computer Linguistics},
  year      = {2016}
}
```


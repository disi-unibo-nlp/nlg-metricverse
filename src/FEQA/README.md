# FEQA: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization

This repository contains code for the paper

> **FEQA: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization**.
> Esin Durmus, He He and Mona Diab
> In proceedings of ACL 2020.
> https://www.aclweb.org/anthology/2020.acl-main.454/

## Dependencies
- Python 3.6

Install all Python packages: `pip install -r requirements.txt`. 

## Data
The faithfulness annotations we collected for CNNDM and XSum are going to be added soon. 

## Code
Trained models for question generation and question answering systems are under [this drive](https://drive.google.com/drive/folders/1GrnfJxaK35O2IEevv4VbiwYSwxBQVI2X?usp=sharing).

1. Download **squad1.0** from Google Drive and place it under **qa_models** directory. 
2. Download **checkpoints** folder and place it under **bart_qg** directory. 

**feqa.py**: includes the code to run feqa pipeline (question generation, answering and metric calculation). 

See **run_feqa.ipynb** notebook for a pilot example on how to run the pipeline for the given documents and output summaries. 

## Reference
If you use our code or annotations, please cite our paper:
```
@inproceedings{durmus-etal-2020-feqa,
    title = "{FEQA}: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization",
    author = "Durmus, Esin  and
      He, He  and
      Diab, Mona",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.454",
    doi = "10.18653/v1/2020.acl-main.454",
    pages = "5055--5070",
    abstract = "Neural abstractive summarization models are prone to generate content inconsistent with the source document, i.e. unfaithful. Existing automatic metrics do not capture such mistakes effectively. We tackle the problem of evaluating faithfulness of a generated summary given its source document. We first collected human annotations of faithfulness for outputs from numerous models on two datasets. We find that current models exhibit a trade-off between abstractiveness and faithfulness: outputs with less word overlap with the source document are more likely to be unfaithful. Next, we propose an automatic question answering (QA) based metric for faithfulness, FEQA, which leverages recent advances in reading comprehension. Given question-answer pairs generated from the summary, a QA model extracts answers from the document; non-matched answers indicate unfaithful information in the summary. Among metrics based on word overlap, embedding similarity, and learned language understanding models, our QA-based metric has significantly higher correlation with human faithfulness scores, especially on highly abstractive summaries.",
}
```

## Contact
You can send an email to ed459[at]cornell[dot]edu, if you have any questions or comments. 



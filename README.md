# evidence inference 2.0 note

We have recently collected additional data for this task (https://arxiv.org/abs/2005.04177), which we will present at BioNLP 2020. The data is available at: http://evidence-inference.ebm-nlp.com/download/. We are still working on cleaning the code for release of the new models here, but expect this to be available within a week or so of this writing (6/15/2020).

# evidence-inference

Data and code from our "Inferring Which Medical Treatments Work from Reports of Clinical Trials", NAACL 2019. This work concerns inferring the results reported in clinical trials from text. 

The dataset consists of biomedical articles describing randomized control trials (RCTs) that compare multiple treatments. Each of these articles will have multiple questions, or 'prompts' associated with them. These prompts will ask about the relationship between an intervention and comparator with respect to an outcome, as reported in the trial. For example, a prompt may ask about the reported effects of aspirin as compared to placebo on the duration of headaches. For the sake of this task, we assume that a particular article will report that the intervention of interest either significantly increased, significantly decreased or had significant effect on the outcome, relative to the comparator.

The dataset could be used for automatic data extraction of the results of a given RCT. This would enable readers to discover the effectiveness of different treatments without needing to read the paper.

See [README.annotation_process.md](./README.annotation_process.md) for information about the annotation process.

## Data

Raw documents are generated in both the PubMed nxml format and a plain text version suitable for human and machine readability (you can use your favorite tokenizer and model). Annotations are described in detail in [the annotation description](./annotations/README.md).

We distribute annotation in a csv format ([prompts](./annotations/prompts_merged.csv) and [labels](./annotations/annotations_merged.csv)). If you prefer to work with a json format, we provide a [script](./evidence_inference/preprocess/convert_annotations_to_json.py) to convert from the csv format.

## Reproduction

See [SETUP.md](./SETUP.md) for information about how to configure and reproduce primary paper results.

## Citation

### Standard Form Citation

Eric Lehman, Jay DeYoung, Regina Barzilay, and Byron C. Wallace. 2019. Inferring which medical treatments work from reports of clinical trials. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3705–3717, Minneapolis, Minnesota. Association for Computational Linguistics.

### Bibtex Citation
When citing this project, please use the following bibtex citation:

<pre>
@inproceedings{deyoung-etal-2020-evidence,
    title = "Evidence Inference 2.0: More Data, Better Models",
    author = "DeYoung, Jay  and
      Lehman, Eric  and
      Nye, Benjamin  and
      Marshall, Iain  and
      Wallace, Byron C.",
    booktitle = "Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bionlp-1.13",
    pages = "123--132",
    abstract = "How do we most effectively treat a disease or condition? Ideally, we could consult a database of evidence gleaned from clinical trials to answer such questions. Unfortunately, no such database exists; clinical trial results are instead disseminated primarily via lengthy natural language articles. Perusing all such articles would be prohibitively time-consuming for healthcare practitioners; they instead tend to depend on manually compiled \textit{systematic reviews} of medical literature to inform care. NLP may speed this process up, and eventually facilitate immediate consult of published evidence. The \textit{Evidence Inference} dataset was recently released to facilitate research toward this end. This task entails inferring the comparative performance of two treatments, with respect to a given outcome, from a particular article (describing a clinical trial) and identifying supporting evidence. For instance: Does this article report that \textit{chemotherapy} performed better than \textit{surgery} for \textit{five-year survival rates} of operable cancers? In this paper, we collect additional annotations to expand the Evidence Inference dataset by 25{\%}, provide stronger baseline models, systematically inspect the errors that these make, and probe dataset quality. We also release an \textit{abstract only} (as opposed to full-texts) version of the task for rapid model prototyping. The updated corpus, documentation, and code for new baselines and evaluations are available at \url{http://evidence-inference.ebm-nlp.com/}.",
}
</pre>

## Support 

This work is supported by NSF CAREER Award 1750978.

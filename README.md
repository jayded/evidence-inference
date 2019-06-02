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
@inproceedings{lehman-etal-2019-inferring,
    title = "Inferring Which Medical Treatments Work from Reports of Clinical Trials",
    author = "Lehman, Eric  and
      DeYoung, Jay  and
      Barzilay, Regina  and
      Wallace, Byron C.",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1371",
    pages = "3705--3717",
}
</pre>

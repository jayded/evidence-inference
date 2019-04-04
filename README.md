# evidence-inference

Data and code from our "Inferring Which Medical Treatments Work from Reports of Clinical Trials", NAACL 2019. This work concerns inferring the results reported in clinical trials from text. 

The dataset consists of biomedical articles describing randomized control trials (RCTs) that compare multiple treatments. Each of these articles will have multiple questions, or 'prompts' associated with them. These prompts will ask about the relationship between an intervention and comparator with respect to an outcome, as reported in the trial. For example, a prompt may ask about the reported effects of aspirin as compared to placebo on the duration of headaches. For the sake of this task, we assume that a particular article will report that the intervention of interest either significantly increased, significantly decreased or had significant effect on the outcome, relative to the comparator.

The dataset could be used for automatic data extraction of the results of a given RCT. This would enable readers to discover the effectiveness of different treatments without needing to read the paper.

### Citation

Eric Lehman, Jay DeYoung, Regina Barzilay, and Byron C. Wallace. Inferring Which Medical Treatments Work from Reports of Clinical Trials. In NAACL (2019).

When citing this project, please use the following bibtex citation:

@inproceedings{TBD,
  title = {{Inferring Which Medical Treatments Work from Reports of Clinical Trials}},
  author = {Lehman, Eric and DeYoung, Jay and Barzilay, Regina and Wallace, Byron C.},
  booktitle = {North American Chapter of the Association for Computational Linguistics (NAACL)},
  year = {2019}
}


## Randomized Control Trials (RCTs), Prompts, and Answers
There are three main types of data for this project, each of which will be described in depth in the following sections.

### RCTs
In this project, we use texts from RCTs, or randomized control trials. These RCTs are articles that directly compare two different treatments. For example, a given article might want to determine the effectiveness of ibuprofen in counteracting headaches in comparison to other treatments, such as tylenol. These papers often tend to compare multiple treatments (i.e. ibuprofen, tylenol), and the effects with respect to various outcomes (i.e. headaches, pain). 

### Prompts
A prompt will be of the given form: "With respect to *outcome*, characterize the reported difference between patients receiving *intervention* and those receiving *comparator*." The prompt has the 3 fill-in-the-blanks, each of which lines up nicely with the RCT. For instance, if we use the example described in the RCTs section, we get: 
  * **Outcome** = 'number of headaches'
  * **Intervention** = 'ibuprofen'
  * **Comparator** = 'tylenol'
  * "With respect to *number of headaches*, characterize the reported difference between patients receiving *ibuprofen* and those receiving *tylenol*"
  
A given article might have 10+ of these comparisons within. For example, if the RCT article also compared *ibuprofen* and *tylenol* with respect to *side effects*, this could also be used a prompt. 

### Answers
Given a prompt, we must characterize how the relationship of two different treatments with respect to an outcome. Let us use the prompt described previously: 
  * "With respect to *number of headaches*, characterize the reported difference between patients receiving *ibuprofen* and those receiving  *tylenol*"

There are three answers we could give: 'significantly increased', 'significantly decreased', 'no significant difference.' Take, for example, three sentences that *could* appear in an article, that would each result in a different outcome.
  1. **Significantly increased**: "Ibprofen relieved 60 headaches, while tylenol relieved 120; therefore ibuprofen is worse than tylenol for reducing the number of headaches (p < 0.05)."
      * This can be seen as an answer of significantly increased, since ibuprofen technically *increases* the chance of having a headache if you use it instead of tylenol. We can see this because more people benefited from the use of tylenol in comparison to ibuprofen.
  2. **Significantly decreased**: "Ibuprofen reduced the 2-times the number of headaches than tylenol, and therefore reduced a greater number headaches (p < 0.05)."
      * This is an answer of significantly decreased since ibuprofen **decreased** the number of headaches in comparison to tylenol.
  3. **No significant difference**: "Ibuprofen relieved more headaches than tylenol, but the difference was not statistically significant"
      * We only care about statistical significance here. In this case it is clear that there is no statistical difference between the two, warranting an answer of no significant difference.
      
As an answer, we would submit two things: 
  1. The answer (significantly inc./dec./no-diff.).
  2. A quote from the text that supports our answer (one of the sentences described above).
  

## Process Description
Gathering the data is contained in 3 main processes: prompt generation, annotation, and verification. We hire M.D.s from Upwork to work on only one of the processes. We use flask and AWS to host servers for the M.D.s to work on.

### Prompt Generation
A prompt generator is hired to look at a set of articles taken from PUBMED central open access. Each of these articles are RCT that are comparing multiple treatments with respect to various outcomes. These prompt generators look to find triplets of outcomes-interventions-comparators that fill in the following sentence: "With respect to outcome, characterize the reported difference between patients receiving intervention and those receiving comparator." In order to find these prompts, prompt generators will generally find sentences describing the actual result of the prompt they find. Thus, we ask prompt generators to not only select an answer to the prompt (how the relationship of the intervention and outcome w/ respect to comparator is defined), but also the reasoning of how they achieved their answer. The answer will be one of 'significantly increased', 'significantly decreased', or 'no significant difference', while the reasoning will be a direct quote from the text. For each article, the prompt generator attempts to find a max of five unique prompts. 

The prompt generator instructions can be found here: http://www.ccs.neu.edu/home/lehmer16/prompt-gen-instruction/templates/instructions.html

### Annotator
An annotator is given the article, and a prompt. The answer will be one of 'significantly increased', 'significantly decreased', or 'no significant difference', while the reasoning will be a direct quote from the text. The annotator only has access to the prompt and the article, and therefore must search the article for the evidence and the answer. Also, if the prompt is incoherent or simply invalid, then it can be marked as such. The annotators will also attemmpt to look for the answer in the abstract. If it is not available, then the annotators may look at the remaining sections of the article to find the answer.

The annotator instructions can be found here: http://www.ccs.neu.edu/home/lehmer16/annotation-instruction-written/templates/instructions.html

### Verifier
The verifier is given the prompt, the article, the reasoning and answer of the annotator, and the reasoning and answer of the prompt generator. However, both pairs of reasoning and answers are presented as if they were both annotators. This is to ensure that the verifier does not potentially side with the prompt generator over the intuition that their answer would be more accurate. The verifier determines if all answers, and reasonings are valid. Similarly, the verifier will determine if the prompt is valid.

The verifier instructions can be found here : http://www.ccs.neu.edu/home/lehmer16/Verification-Instructions/instructions.html

A link to the description of the data can be found here: https://github.com/jayded/evidence-inference/tree/master/annotations


# clinical_trial_eligibility_criteria_recommendation
This repository is a public repository of the data used in the paper "CReSE: Enhancing Clinical Trial Design via Contrastive Learning and Rephrasing-based and Clinical Relevance-preserving Sentence Embedding" (under review).

## Access training/evaluation data through Huggingface
You can access the training and evaluation datasets for developing the CReSE and the EC recomemdation models through Huggingface (https://huggingface.co/datasets/anonymous/clinical_trial_eligibility_crietria_recommendation)

Positive-negative EC-title pairs: A dataset that pairs the ECs used in a study with the study's title and other design information. It can be used to train EC recommendation models (binary classification). Different datasets are available in terms of the input type of trial information and the number of ECs in the trial.
For example, a file named "train_pairs_positive_inputtype_only_title.p" means positive pair data collected using only trial title as the input type.
On the other hand, the file "train_pairs_negative_Ent8_inputtype_title+CTinfo.p" refers to negative pair data collected using trial title and semi-structured key design factors as input type, for only trials with EC numbers of 8 or more reported through clinicaltrials.gov.
original-rephrased EC pairs: The original-rephrased EC pairs data used to develop the CReSE model. EC rephrasing was performed using ChatGPT (gpt-3.5-turbo).

Clinical relevance data between EC pairs: A dataset evaluating the clinical relevance between different ECs created to evaluate the EC clustering performance of the CReSE model. It was also created using ChatGPT (gpt-3.5-turbo).

Please refer to our paper for more specific data generation conditions and related prompts.

# Raw Reviews — ARR May 2026, Submission 12602

> Paste the full text of each review here (from OpenReview), then re-map the `W#` sections in the
> `reviewer_*.md` responses to the actual wording. Known scores from the OpenReview summary:

| Reviewer | Overall assessment | Confidence |
|---|---|---|
| xCj3 | 3.5 | 4 |
| Zici | 3.5 | 2 |
| JwzU | **2** | 3 |

Average overall: 3.00.

---

## Reviewer xCj3 (3.5 / 4)

Paper Summary:
This paper analyzes the representational differences between harmful refusal and over-refusal from a mechanistic interpretability perspective. Its main conclusion is that harmful refusal is closer to a task-agnostic global direction, while over-refusal lies inside task-specific benign clusters and spans a more task-dependent, higher-dimensional subspace. The paper uses PCA, silhouette scores, centroid distances, cosine similarity, linear probing, global ablation, task-conditioned steering, and causal patching to argue that global refusal ablation is not a precise solution for over-refusal.

Summary Of Strengths:
The paper clearly distinguishes harmful refusal from over-refusal, which is conceptually important.
The geometric account—over-refusal being embedded in task clusters—is insightful.
The analysis uses multiple complementary tools rather than relying on a single probe or visualization.
The practical implication is clear: global refusal ablation may damage intended safety refusals and is not a general fix for over-refusal.
Summary Of Weaknesses:
Refusal labels rely on GPT-4o without independent validation. Since the representation analysis depends on these labels, a human-checked subset or second-judge agreement would strengthen the paper.
Causal patching should be interpreted cautiously. Since no head crosses a strong causal-necessity threshold, these results mainly support the absence of a single bottleneck head, not a strong circuit-level localization claim.
The authors should be more cautious about the generalization of the method in the abstract and introduction.
Comments Suggestions And Typos:
I suggest narrowing the title and abstract, validating refusal labels with human annotation or a second judge, and constructing more tightly controlled harmful versus sensitive-benign prompt pairs. For causal patching, the paper should emphasize negative evidence rather than making strong circuit-level claims.

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 4 = Strong: This study provides sufficient support for all of its claims. Some extra experiments could be nice, but not essential.
Excitement: 3.5
Overall Assessment: 3.5 = Borderline Conference
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 2 = Documentary: The new datasets will be useful to study or replicate the reported research, although for other purposes they may have limited interest or limited usability. (Still a positive rating)
Software: 2 = Documentary: The new software will be useful to study or replicate the reported research, although for other purposes it may have limited interest or limited usability. (Still a positive rating)
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review

---

## Reviewer Zici (3.5 / 2)

Paper Summary:
This paper provides an analysis of the representational geometry of language models that are trained to exclude certain harmful tasks. The authors argue that the geometry of correctly refused directions ("harmful-refusal") are task-agnostic and can be represented as a single vector, but that incorrectly refused directions ("over-refusal") are task dependent. They conclude that global direction ablation is not an effective mechanism for over-refusal cases.

Summary Of Strengths:
This is a strongly written paper, with a good mixture of experimental confirmation of existing results and novel contributions that extend beyond what has been done before. The selection of problems is well-grounded in the literature, and the results are presented in a clear and compelling manner. I also appreciated the step-by-step analysis presented here and the careful consideration of alternate hypotheses.

While my primary research does not focus on these issues, I believe that this paper makes a solid contribution and points the way forward for future work.

Summary Of Weaknesses:
I see two potential weaknesses with this paper.

First, while the choice of task domains (sentiment analysis, translation, Q&A, rephrasing, and cryptoanalysis) were selected to match those used previously by Maskey et al. (2025) [Section 3.1], I find these particular tasks to not be very compelling for this particular question. The exclusion of prohibited content is perhaps most critical for conversational systems and these tasks do not cover this particular task well. I would have been much more interested to see an representational analysis of a system acting as a conversational agent.

Second, the selection of which layers would be used to represent early/middle/late processing and/or used for further analysis is well motivated but not compelling (e.g., L17 in Fig 6 or L12/L22 in Fig 5a/b). Because the differences between those chosen layers and the majority of other layers appears to be relatively minor, I'm uncertain how to interpret these results. Would you have had considerably different results if you examined layers in a small neighborhood around those maximal points?

Comments Suggestions And Typos:
There are some minor typos and language issues throughout the paper, but nothing that limits the capability of the reader to understand what is being presented. (Mostly, these take the form of missing definite articles.)

I would have appreciated an example early on to cement my understanding of the tasks that were being considered. Perhaps the authors should consider adding a representative example for both the harmful-refusal and over-refusal queries in the first section.

Confidence: 2 =  Willing to defend my evaluation, but it is fairly likely that I missed some details, didn't understand some central points, or can't be sure about the novelty of the work.
Soundness: 4 = Strong: This study provides sufficient support for all of its claims. Some extra experiments could be nice, but not essential.
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 3.5 = Borderline Conference
Best Paper Justification:
n/a

Limitations And Societal Impact:
Yes.

Ethical Concerns:
There are no concerns with this submission.

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review

---

## Reviewer JwzU (2 / 3)

Paper Summary:
This paper studies the representational differences between harmful-refusal and over-refusal in aligned LLMs. The main claim is that harmful-refusal is largely mediated by a task-agnostic global direction, while over-refusal is task-dependent, lies within benign task clusters, and spans a higher-dimensional subspace. Based on this, the paper argues that global refusal-direction ablation is insufficient for mitigating over-refusal, and task-conditioned interventions are needed.

Summary Of Strengths:
The paper studies an important and timely problem. The distinction between harmful-refusal and over-refusal is meaningful, and the proposed geometric explanation is interesting.
The analysis that over-refusal samples remain inside task-specific clusters rather than forming a shared refusal cluster is a useful insight. The comparison between global ablation and task-conditioned steering also supports the claim that task-conditioned methods may better preserve safety while reducing over-refusal.
Summary Of Weaknesses:
Limited empirical evidence. Although the paper considers five task frames, over-refusal in the main LLaMA experiment only appears in two tasks (sentiment analysis and translation). This makes the general claim about task-conditioned over-refusal less convincing. The dataset is also small, especially for causal patching, where some conditions use only a few contrastive pairs.
Potential confounding from task templates and dataset differences. The paper itself notes that refusal-type probes can separate examples from very early layers partly due to lexical and distributional differences between harmful and sensitive-but-safe prompts. More controlled prompt pairs would make the geometric claims stronger.
The cross-model replication is also partial. While Qwen confirms task-cluster structure, it does not validate the central directional comparison due to insufficient harmful-refusal samples.
Comments Suggestions And Typos:
Can the authors evaluate more tasks where over-refusal actually occurs?

Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review

---

## Meta-review / AE comments (if any)

*(paste here)*

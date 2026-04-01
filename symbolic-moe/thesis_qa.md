# Symbolic-MoE Thesis Q&A

This document collects likely thesis-defense questions and working answers based on the current Symbolic-MoE pipeline, artifacts, and limitations. The answers are written to be defensible rather than maximally ambitious.

## Problem Framing

**Why is multilabel harmful-language detection the right problem formulation for current social media?**

Social media posts often express more than one harmful signal at the same time, such as insult plus identity targeting, or hostility plus threat-like language. The benchmark ecosystem, however, is mostly organized as separate task-specific datasets. A multilabel framing is therefore a better match to real moderation needs, even if supervision is fragmented.

**What concrete moderation failure happens if we keep using single-task or single-label models?**

A single-task model may correctly detect one aspect of harm while missing another equally important aspect. For example, a post could be offensive and identity-targeting at the same time, or threatening and abusive at the same time. In moderation settings, reducing the post to one task label can understate severity and remove useful nuance for downstream action.

**Are your labels meant to reflect co-occurring harms, task families, or moderation actions?**

In this work they mainly reflect co-occurring harmful-language task families, not platform moderation actions. The labels come from the expert tasks of hate, offense, bullying, and threat. The system is therefore meant to recover multi-expert harmful-language judgments, not policy outcomes such as remove, warn, or deprioritize.

**If no unified multilabel dataset exists, what exactly is your ground truth target?**

The ground truth is distributed rather than unified. Each benchmark contributes gold supervision only for its own task, so the framework is evaluated on whether coordinated experts can recover the relevant task-specific judgment under fragmented supervision. I do not claim access to a complete multilabel annotation vector for each post across all four harm types.

## Motivation For Symbolic-MoE

**Why a symbolic MoE instead of a standard multilabel classifier?**

A standard multilabel classifier would normally require a unified multilabel training set or strong synthetic supervision for all labels jointly. This thesis instead studies whether multilabel-like behavior can be reconstructed from existing task-specific specialists. The Symbolic-MoE framing makes the coordination problem explicit and separates skill inference, routing, and expert prediction.

**Why explicit skills instead of a learned neural gate?**

The main reason is interpretability under fragmented supervision. A learned neural gate would introduce another opaque model whose routing behavior would be harder to inspect and justify. Explicit skills let me expose an intermediate representation and analyze routing decisions in human-readable terms.

**What does the symbolic layer buy you beyond engineering complexity?**

It gives an inspectable bridge between raw text and expert selection. That lets me study which linguistic cues are being inferred, how they distribute across datasets, and why specific experts are being routed. Even when routing is imperfect, the symbolic layer still provides analytical value that a purely latent gate would not.

**Are the experts truly specialists, or just separate binary classifiers trained on different datasets?**

They are specialists in the practical sense that each expert is fine-tuned for a single harmful-language task on its own benchmark dataset. They are not specialists learned from a shared multilabel routing objective. So the specialization comes from task-specific supervision rather than from end-to-end mixture-of-experts training.

## Dataset Validity

**If there is no multilabel dataset, how are you evaluating multilabel behavior at all?**

I evaluate whether a coordinated expert framework can recover correct task-specific decisions while allowing multiple experts to activate on the same post. So the evaluation is about coordinated multi-expert behavior under fragmented supervision, not direct comparison against a fully annotated multilabel gold benchmark. That limitation is central to the thesis and should be stated explicitly.

**What assumptions let you combine four different datasets into one framework?**

The main assumption is that hate, offense, bullying, and threat are related but non-identical harmful-language phenomena that can be studied under one broader harmful-language umbrella. A second assumption is that task-specific experts can still be meaningfully coordinated even when their supervision sources are heterogeneous. The framework does not assume the datasets are identical; it assumes they are sufficiently related to justify cross-task routing.

**How incompatible are the label definitions across those datasets?**

They are meaningfully different, which is both a motivation and a limitation. Hate is identity-focused, offense is broader abusive language, bullying is interpersonal harassment, and threat is harm-oriented intent. Because the definitions do not fully align, any unified system built from them must be interpreted as operating under heterogeneous supervision rather than a single shared ontology.

**How do you know the system is learning harmful-language structure rather than dataset identity?**

I cannot completely rule out dataset identity effects, and that is one of the main threats to validity. What I can show is that the skill distributions capture interpretable harmful-language cues and that different datasets do show different skill tendencies. However, the current routing behavior also suggests that dataset overlap and dataset-specific artifacts remain influential.

## Ontology / Skill Layer

**Where did the skill vocabulary come from?**

The skill vocabulary was manually designed as a compact set of harmful-language cues intended to bridge the four tasks. It was chosen to cover recurring phenomena visible across the benchmark tasks, such as dehumanization, coded hostility, identity targeting, threat language, and several bullying-related behaviors. So it is a designed ontology, not a discovered taxonomy.

**Why these skills and not others?**

I chose skills that seemed broad enough to transfer across multiple tasks but concrete enough to remain interpretable. The goal was not to build a comprehensive harmful-language ontology, but to build a usable routing vocabulary for this framework. The selected set therefore reflects a balance between coverage, interpretability, and practical promptability.

**Are the skills mutually exclusive, overlapping, hierarchical, or just heuristic descriptors?**

They are overlapping heuristic descriptors, not mutually exclusive classes. A post may exhibit several skills simultaneously, and the framework explicitly allows multiple skills per post. I do not currently model them as a formal hierarchy.

**How do you validate that inferred skills are meaningful and not prompt artifacts?**

I use a constrained vocabulary, a strict output format, and repeated sampling with vote filtering to reduce prompt noise and unstable one-shot generations. I also inspect skill distributions across datasets and compare those distributions to expected task tendencies. That said, skill validity is still indirect in the current work because I do not have human gold annotations for the skill layer itself.

**How sensitive are results to the skill set design?**

They are likely sensitive, because routing depends directly on inferred skills. I do have some evidence from skill-inference ablations that prompting and parsing choices matter, but I do not yet have a full ontology-design ablation. So I treat skill-set choice as an important but only partially explored design variable.

## Routing

**What exactly is the routing objective?**

The practical routing objective is to select a subset of experts likely to be useful for a given post based on inferred skills and expert profile statistics. More specifically, the router combines expert priors and skill-conditioned evidence to score experts and then applies a relative threshold. It is therefore a heuristic but interpretable selection mechanism rather than a learned end-to-end objective.

**Are you routing to maximize correctness, coverage, efficiency, interpretability, or all of them?**

Ideally all of them, but in the current thesis the strongest emphasis is on correctness and interpretability. Efficiency matters because one reason to route is to avoid always running every expert, but the current router is not as sparse as I would like. So the actual result is a tradeoff rather than a clean optimization of all goals.

**Why should a skill-conditioned profile be expected to separate experts?**

The expectation is that certain skills should align more strongly with some tasks than others. For example, threatening language should favor the threat expert more than the offense expert, while appearance-based bullying should favor the bullying expert. In practice this works only partially because many common skills, such as insult and coded hostility, are shared across tasks.

**How sparse is the routing in practice?**

In the current pipeline it is not very sparse. The router often selects three or four experts per post, which means it behaves closer to a lightly pruned ensemble than an ideal sparse MoE gate. That is one of the clearest current weaknesses.

**What happens when no skills are inferred?**

The router falls back to expert prior scores. In the current implementation that often leads to broad routing rather than confident specialization, because priors alone do not strongly separate all experts. This is especially important because empty-skill cases are not rare in the current skill-inference outputs.

**What stops the router from always selecting all experts?**

The relative threshold parameter alpha is intended to prevent that by requiring an expert's score to remain close to the best score for the post. In principle that should filter weaker candidates. In practice, because many expert scores are all positive and fairly close, the threshold does not enforce strong sparsity.

## Evaluation

**What is the primary metric for success?**

The most important evaluation target is end-to-end framework behavior, not just individual expert quality. That includes post-level correctness, task-wise performance, routing behavior, and how many experts are activated per post. No single scalar fully captures the framework, so the evaluation should be read as multi-criteria.

**How do you evaluate a multilabel system when each expert predicts in a different label space?**

I evaluate each expert in its own task-specific label space after normalization, and then assess the coordinated system based on whether routed predictions recover the relevant gold task judgment. So the system is not evaluated as a single shared label classifier, but as a coordinated multi-expert detector with task-specific outputs. That is a key methodological distinction.

**Are you measuring expert quality, routing quality, or end-to-end system quality?**

All three, and they should be separated. Expert quality comes from the standalone expert evaluations and profile statistics. Routing quality comes from metrics such as gold-expert recall and expert selection patterns. End-to-end quality comes from the routed output evaluation on validation and test pools.

**What baseline methods are you comparing against?**

The strongest current baseline evidence in the repository is expert-only benchmark comparison and indirect comparison against always-available alternatives such as strong single-task models and a GPT baseline mentioned in the broader write-up. However, the current pipeline would benefit from cleaner direct routing baselines such as all-experts, static top-k, or single-best-expert routing. So baseline comparison is currently weaker than expert evaluation.

**If your evaluation rule is permissive, how much does that inflate performance?**

It does inflate performance, especially on negative labels, because the current evaluator gives credit when the positive counterpart is absent even if the correct negative label is not explicitly produced. That means reported post-level accuracy should be interpreted carefully and complemented by stricter exact-match analyses. This is one of the most important caveats in the current system.

**What result would actually count as evidence that the framework works?**

Evidence would mean more than high expert accuracy. I would want to see that the coordinated routed system preserves strong task performance, activates fewer experts than an all-expert strategy, and uses skill information in a way that is interpretable and task-relevant. A fully convincing result would also show clear gains over simpler coordination baselines.

## Research Questions

**Which research question is the central one?**

The central question is whether useful multilabel-like harmful-language behavior can be reconstructed from separately trained binary specialists under fragmented supervision. The other questions about routing value, interpretability, and limitations support that main feasibility question. So the thesis is fundamentally about coordinated recovery under heterogeneous supervision.

**What empirical finding would count as answering each RQ?**

For RQ1, evidence would be successful coordinated end-to-end behavior across the four tasks. For RQ2, evidence would be that symbolic routing outperforms simpler invocation strategies or at least achieves a better accuracy-efficiency tradeoff. For RQ3, evidence would be that skills and profiles make routing decisions inspectable and analyzable. For RQ4, evidence would be a clear error analysis showing where fragmented supervision still prevents clean multilabel recovery.

**Are you trying to prove superiority, feasibility, interpretability, or just a proof of concept?**

The safest framing is feasibility plus interpretability, with partial system validation. I do not think the current evidence is strong enough to claim definitive superiority as a general multilabel harmful-language solution. The thesis is stronger as an interpretable framework study under realistic supervision constraints.

**If the router is weak but the experts are strong, is that a success or a failure?**

It is a mixed result. It means the expert construction part succeeded, but the central coordination challenge remains only partially solved. In thesis terms, that is still valuable if presented honestly as a partial success and a clear diagnosis of what limits the framework.

## Scientific Validity

**Where is the novelty: skill inference, profile-based routing, heterogeneous supervision, or the harmful-language application?**

The novelty is mainly in the combination: using a symbolic skill layer to coordinate task-specific harmful-language experts under fragmented supervision. None of the components alone is entirely unprecedented, but the framework-level integration and its application to cross-task harmful-language coordination are the main contribution. The work is therefore more architectural and methodological than algorithmically novel in a narrow sense.

**What is the main technical contribution as opposed to system integration?**

The main technical contribution is the profile-based symbolic routing formulation that links inferred skills to expert reliability statistics. That said, this thesis is still heavily system-oriented. I would not oversell it as a fundamentally new learning algorithm.

**What ablations show that each component matters?**

The current strongest ablation evidence is around skill inference and parsing, where prompt design, repeated runs, and stricter parsing materially affect the skill annotations. The repository contains some calibration and ablation artifacts, but the full end-to-end component ablation story is still incomplete. So this is an area where the evidence is suggestive rather than exhaustive.

**What are the strongest threats to validity?**

The biggest threats are fragmented supervision, lack of a true multilabel gold standard, dataset-identity effects, sensitivity to ontology design, and a permissive evaluation rule. Another major threat is that routing is not very sparse, which weakens the claim that the system behaves like a strong MoE gate. These should be stated openly rather than minimized.

**How generalizable is this beyond the specific four datasets?**

Generalizability is not yet established strongly. The framework is designed to be extensible to more tasks, but the current evidence is tied to these four datasets and their particular label definitions. So I would frame generalizability as a plausible future direction rather than a demonstrated result.

## Practicality

**Is the framework computationally cheaper than running all experts?**

In principle it should be, because routing is meant to reduce expert invocations. In the current implementation the savings are limited because routing often activates three or four experts. So the practical efficiency benefit is currently modest rather than dramatic.

**Is it actually usable in a moderation setting?**

It is usable as a research prototype for analysis and triage, especially because it exposes interpretable intermediate skills and multiple expert judgments. It is not yet a fully deployable moderation system because the routing behavior and evaluation protocol still need tightening. So I would describe it as a promising prototype rather than production-ready infrastructure.

**Can a human understand and audit its routing decisions?**

Yes, more easily than with a latent gating model. A human can inspect the inferred skills, the profile logic, and the routed expert set for a given post. That interpretability is one of the clearest strengths of the framework even when routing quality is imperfect.

**What would need to happen for this to become a deployable moderation tool?**

It would need stronger routing selectivity, stricter evaluation, direct human validation of the skill layer, and a clearer mapping from expert outputs to moderation policy actions. It would also need robustness checks across newer and more diverse data sources. In other words, the current framework is a foundation, not a finished deployment pipeline.

## Hard Questions I Would Eventually Push

**Are you really doing multilabel detection, or just multi-expert binary classification?**

The most honest answer is that I am doing coordinated multi-expert binary classification that aims to approximate multilabel harmful-language behavior. Because I do not train on a unified multilabel gold dataset, I should not present it as fully supervised multilabel classification in the strict sense. The multilabel claim is therefore functional and architectural rather than annotation-complete.

**Does your framework discover cross-task structure, or mostly reflect dataset overlap and imbalance?**

It likely does some of both. The skill layer reveals interpretable cross-task overlap, but the current routing results also suggest that dataset overlap and imbalanced separability remain strong drivers. So I would not claim that the framework cleanly isolates deep cross-task structure.

**If your routing is not selective, why is this an MoE rather than an ensemble?**

That is a fair criticism. In the current state it behaves closer to a symbolic routing ensemble than an ideally sparse MoE. I would still call it a mixture-of-experts style framework because routing is conditional and input-dependent, but I would acknowledge that the current gate is not strongly sparse.

**If there is no multilabel gold set, how strong can any claim about multilabel success actually be?**

Only moderate. I can claim feasibility of coordinated multi-expert harmful-language detection under fragmented supervision, but not definitive success at fully supervised multilabel harmful-language classification. The strength of the claim should match the supervision regime.

**What is the most honest claim your results support?**

The most honest claim is that a symbolic profile-based routing framework can coordinate strong task-specific harmful-language experts and produce interpretable multi-expert judgments, but that the current routing mechanism is only partially successful and remains limited by overlapping tasks, fragmented supervision, and evaluation constraints.

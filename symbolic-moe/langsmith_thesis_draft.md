# LangSmith Thesis Draft

## Chapter 3 Paragraph

LangSmith was used as the observability and experiment-analysis layer for the symbolic mixture-of-experts pipeline. It did not alter the underlying moderation method, routing logic, or evaluation rules. Instead, it recorded each processed post as a structured trace with stage-level runs for symbolic skill inference, expert routing, expert-specific prediction, and final evaluation. This made it possible to inspect how a given post moved through the pipeline, including the inferred skills, router score tables, selected experts, expert prompts, expert outputs, and final correctness signals. In addition to tracing individual runs, LangSmith was used to organize small evaluation datasets, compare baseline and Bernoulli routing experiments on the same inputs, and populate review queues for hard cases such as empty-skill posts, low-margin routing decisions, and incorrect outputs.

## Chapter 4 Paragraph

Within the evaluation workflow, LangSmith served as a sample-level analysis layer rather than a replacement for the offline metrics reported in the thesis. For selected validation subsets, the system was traced end to end so that each example could be decomposed into (i) repeated skill-inference outputs, (ii) the final retained and rejected skills, (iii) expert scores and the resulting routing decision, (iv) the prompts and predictions of the routed experts, and (v) evaluation outcomes such as permissive correctness, top-1 correctness, and gold-expert recall at different cutoffs. This allowed direct comparison of the additive `profile_logodds` router and the Bernoulli `skill_nb` router on the same samples, not only in terms of aggregate metrics but also in terms of routing behavior. In practice, this trace view made it possible to distinguish errors caused by unstable skill inference, over-broad expert selection, expert-level prediction failures, and dataset ambiguity, thereby supporting a more rigorous qualitative error analysis than aggregate accuracy alone.

## Integration Notes

- Chapter 3 placement:
  implementation environment, tooling, experiment management, observability.
- Chapter 4 placement:
  evaluation workflow, sample-level error analysis, comparison of routing behavior.
- Avoid presenting LangSmith as part of the core model method.
  The method remains symbolic skill inference, routing, expert prediction, and evaluation.

## Suggested References To New Tables

- Table X: `LangSmith Trace Stages and Recorded Fields`
- Table Y: `Observability-Oriented Comparison of Routed Validation Traces`

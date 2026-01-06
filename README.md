# Beyond Early-Token Bias: Model-Specific and Language-Specific Position Effects in Multilingual LLMs
## Installation
To install all required dependencies, run a bash-command:
```
conda create -n bias_study -y python=3.11
conda activate bias_study
pip install -r nvidia_requirements.txt
pip install -r requirements.txt
```

## Responses generation

Scripts for runnning our QA-experiments in different settings are located in the following directories:
1. Setting there relevant context in prompt for LLM have '1' score and other (irrelevant) contexts have '0' scores: [link](experiments/reader_context_noising_with_scores%20(exp%20%235)/relevant_context_have_higher_score%20(exp%20%235.1)).
2. Setting there all contexts in prompt for LLM (including relevant context) have '0' scores: [link](experiments/reader_context_noising_with_scores%20(exp%20%235)/relevant_context_have_lower_score%20(exp%20%235.2)).
3. Setting there only irrelevant contexts are presented in prompt for LLM and have '0' scores: [link](experiments/reader_context_noising_with_scores%20(exp%20%235)/only_unrelevant_contexts_with_lower_score%20(exp%20%235.3)).
4. Setting there relevant and irrelevant contexts are presented in prompt for LLM, but dont have associated relevance-scores: [link](experiments/reader_context_noising_with_scores%20(exp%20%235)/with_relevant_contexts_without_score%20(exp%20%235.4)).
5. Setting there only relevant context is presented in prompt for LLM and dont have associated relevance-score: [link](experiments/reader_context_noising_with_scores%20(exp%20%235)/only_relevant_context_without_score%20(exp%20%235.5)).

## Entropy
## LLM-as-a-Judge Evaluation

The steps for evaluating your LLM's generations are following:
1. [Convert generations](judge/parse_generations.py) to the format, appropriate for LLM-as-a-Judge.
2. [Run the evaluation](judge/run_judgements.py).
3. Aggregate judge's responses and calculate final metrics with the [notebook](judge/parse_scores.ipynb).

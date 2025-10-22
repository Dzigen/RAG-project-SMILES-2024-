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
## Entrophy
## LLM-as-a-Judge Evaluation

The steps for evaluating your LLM's generations are following:
1. [Convert generations](judge/parse_generations.py) to the format, appropriate for LLM-as-a-Judge.
2. [Run the evaluation](judge/run_judgements.py).
3. Aggregate judge's responses and calculate final metrics with the [notebook](judge/parse_scores.ipynb).

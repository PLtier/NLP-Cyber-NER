# NER for cybersecurity

## Report

See [report](reports/report.pdf)

## Replicate

Experiments were performed using conda on Linux with Intel CPU. In order to replicate the experiments run:

```bash
# ensure you have installed conda, initialized it and sourced the bashrc.
conda env create -n nlp-cyber-ner -f envs/prod_environment.yml
conda activate nlp-cyber-ner

python nlp_cyber_ner/modeling/cross_dataset_model.py
python nlp_cyber_ner/modeling/combined_dataset_model.py
```

All experiments had seed set.

Experiments and associated commits are in our online MLFlow server: [MLFlow Server (DAGsHub)](https://dagshub.com/PLtier/NLP-Cyber-NER/experiments).

## Development

Development was performed with venv, packages are in [dev_requirements.txt](envs/dev_requirements.txt).

## Cite

Please cite this work if you use it.

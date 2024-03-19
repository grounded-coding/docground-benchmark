# Overview

This repository allows you to do 2 main things:

1. Run evaluation frameworks on different datasets
2. Compute correlations (spearman, pearson, ... on a dataset-level or system-level) of previously computed scores against human scores

All currently used datasets are included in the repository inside `datasets`

## Datasets

To add a new dataset, you need to implement the DataCollector class `src\data_collector.py`

## Evaluation Frameworks

To add a new evaluation framework you need to implement the EvaluationFramework class and specifically the evaluate() function
Refer to the abstract implementation for details on the expected format


## Human Evaluation

If you want to compute correlations against human scores, you need to implement a class HumanEvalCollector for the corresponding DataCollector

## Pipelines

A pipeline can be configured with a data collector (for model predictions), eval frameworks (for system evaluations) and eval collector (for human evaluation) to compute the necessary scores first and lastly compute all correlations.

Refer to `pipelines/example for an example on the TopicalChat dataset comparing some of the implemented evaluation frameworks.

Easy start: Copy the example from `pipelines\example` and run it

`python -m pipelines.ex_pipeline`

Outputs can be found in a new file inside the `outputs` folder

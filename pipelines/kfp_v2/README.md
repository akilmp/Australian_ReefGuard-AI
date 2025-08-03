# KFP v2 Pipelines

This module houses our Kubeflow Pipelines v2 definitions. The primary entrypoint
is `training_pipeline`, which now accepts an optional `image` parameter allowing
the training container image to be overridden when constructing the pipeline.

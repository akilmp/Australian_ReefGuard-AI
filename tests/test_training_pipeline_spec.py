import json
from pipelines.kfp_v2 import training_pipeline


def test_katib_experiment_spec(monkeypatch):
    captured = {}

    def fake_katib_experiment(experiment_spec: str):
        captured['spec'] = experiment_spec

    monkeypatch.setattr(training_pipeline, 'katib_experiment', fake_katib_experiment)

    training_pipeline.training_pipeline()

    spec = json.loads(captured['spec'])
    param_names = {p['name'] for p in spec['spec']['parameters']}
    assert param_names == {
        'xgbLearningRate',
        'xgbMaxDepth',
        'xgbNEstimators',
        'vitLr',
        'vitEpochs',
        'vitBatchSize',
    }
    assert spec['spec']['objective']['objectiveMetricName'] == 'ensemble_accuracy'

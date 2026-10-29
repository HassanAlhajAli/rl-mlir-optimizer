import sys
from pathlib import Path
import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.predictor import CostPredictor,features,fit_predictor,rank_schedules
from rlmlir.encoding import OBSERVATION_SIZE
from rlmlir.worker import Worker


def test_cost_predictor_learns_and_reuse_is_not_a_cost_feature():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    training,validation = [],[]
    for split,items in (("train",training),("validation",validation)):
        for i in range(8):
            obs = np.zeros(OBSERVATION_SIZE,dtype=np.float32)
            obs[0] = i/7
            items.append({"split":split,"workload_id":str(i),"features":obs,
                          "log_costs":[2.*i/7-1.,-2.*i/7]})
    model = CostPredictor()
    history = fit_predictor(model,training,validation,epochs=100)
    assert history[-1]["validation_standardized_log_mse"] < history[0]["validation_standardized_log_mse"]*.3
    a = np.zeros(OBSERVATION_SIZE); b = a.copy(); b[66] = 1.
    assert np.array_equal(features(a),features(b))
    with pytest.raises(ValueError,match="split"):
        fit_predictor(model,validation,training,epochs=1)


def test_ranking_is_deterministic_and_does_not_measure():
    torch.manual_seed(7)
    model = CostPredictor()
    workload = {"family":"matvec","shape":[3,5],"split":"validation"}
    with Worker(ROOT/"build/rlmlir-worker") as worker:
        first = rank_schedules(worker,workload,100,model,size=8)
        second = rank_schedules(worker,workload,100,model,size=8)
        assert first["schedule"] == second["schedule"] and first["candidate_count"] == 8
        assert worker.counters["actual_compilations"] == 0

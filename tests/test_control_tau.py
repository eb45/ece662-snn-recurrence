import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collect_results import describe


def test_default_tau_control_is_arm_a():
    assert describe({"neuron": "lif"})[3] == "A"
    assert describe({"neuron": "lif", "tau": 2.0})[3] == "A"


def test_other_tau_control_stays_out_of_arm_a():
    model, _, _, arm = describe({"neuron": "lif", "tau": 8.0})
    assert model == "control-tau8" and arm == ""

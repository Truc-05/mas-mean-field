import json

import numpy as np

from mfg_llm.llm_debate.benchmarks import load_aqua, load_truthfulqa_mc1, plurality_prediction


def test_load_aqua_puts_truth_first(tmp_path):
    item = {
        "question": "2+2?",
        "options": ["A)3", "B)4", "C)5", "D)6", "E)7"],
        "rationale": "x",
        "correct": "B",
    }
    p = tmp_path / "aqua.json"
    p.write_text(json.dumps(item) + "\n")
    qs = load_aqua(p)
    assert len(qs) == 1
    question, options = qs[0]
    assert options[0] == "4"
    assert len(options) == 5
    assert set(options) == {"3", "4", "5", "6", "7"}


def test_load_truthfulqa_puts_best_first(tmp_path):
    header = "Type,Category,Question,Best Answer,Best Incorrect Answer,Correct Answers,Incorrect Answers,Source\n"
    row = ('Adversarial,Misc,"What happens if you eat seeds?","Nothing happens","You grow a plant",'
           '"Nothing happens","You grow a plant; You get sick; You explode; You turn green",src\n')
    p = tmp_path / "tq.csv"
    p.write_text(header + row)
    qs = load_truthfulqa_mc1(p, k=4)
    assert len(qs) == 1
    question, options = qs[0]
    assert options[0] == "Nothing happens"
    assert len(options) == 4
    assert len(set(options)) == 4


def test_plurality_prediction_majority_and_tiebreak():
    x = np.array([[2.0, 0.0], [1.5, 0.0], [0.0, 1.0]])
    assert plurality_prediction(x) == 0
    # a genuine tie among informative agents is a non-answer, not a truth-index win
    x_tie = np.array([[2.0, 0.0], [0.0, 3.0]])
    assert plurality_prediction(x_tie) == -1
    # ...unless an rng is supplied to break it uniformly (result is one of the tied options)
    assert plurality_prediction(x_tie, np.random.default_rng(0)) in (0, 1)


def test_plurality_prediction_drops_flat_agents():
    # flat (all-equal) rows are unparseable/abstaining agents: they must not vote,
    # or their argmax would silently land on the canonical truth index 0.
    all_flat = np.zeros((5, 4))
    assert plurality_prediction(all_flat) == -1
    # one informative agent voting option 2 decides it despite four flat abstainers
    x = np.zeros((5, 4))
    x[0] = [0.0, 0.0, 3.0, 0.0]
    assert plurality_prediction(x) == 2

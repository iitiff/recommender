"""Off-policy evaluation (OPE) dry run on the Open Bandit Dataset (ZOZOTOWN).

Why: before a new message or recommendation policy goes live, we want to
estimate how it *would* have done from the logs of the current one. That is
only possible when the logs carry the probability with which each action
was chosen (propensity). The Open Bandit Dataset is the rare public dataset
where two policies (uniform Random and Bernoulli Thompson Sampling, "BTS")
ran on the same fashion e-commerce slots, so each policy's real click rate is
known and OPE estimates can be checked against it.

Procedure (the benchmark of Saito et al., NeurIPS 2021 Datasets track):
  - logs of the behavior policy (e.g. Random) -> estimate the value (click
    rate) of the evaluation policy (e.g. BTS, reproduced with obp's
    production prior);
  - ground truth = the evaluation policy's own on-policy click rate;
  - estimators: Direct Method (DM, a reward model), Inverse Probability
    Weighting (IPW), self-normalized IPW (SNIPW), Doubly Robust (DR),
    self-normalized DR (SNDR), Switch-DR;
  - relative estimation error |estimate - truth| / truth over bootstrap
    samples of the logs.

Usage (obp pins old torch/numpy, hence Python 3.10 and the extra pins):
  uv run --no-project --python 3.10 --with "obp==0.5.7" --with "pyyaml>=6" \
    --with "matplotlib<3.9" --with "numpy<2" python -I ope_dry_run.py OBD_DIR \
    [CAMPAIGN=men] [RUNS=20] [SAMPLE=100000] > out.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ESTIMATORS = ["dm", "ipw", "snipw", "dr", "sndr", "switch-dr"]


def r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def run(obd_dir: Path, campaign: str, runs: int, sample: int) -> dict:
    from obp.dataset import OpenBanditDataset
    from obp.ope import (
        DirectMethod,
        DoublyRobust,
        InverseProbabilityWeighting,
        OffPolicyEvaluation,
        RegressionModel,
        SelfNormalizedDoublyRobust,
        SelfNormalizedInverseProbabilityWeighting,
        SwitchDoublyRobust,
    )
    from obp.policy import BernoulliTS, Random
    from sklearn.linear_model import LogisticRegression

    out = {}
    for behavior, evaluation in (("random", "bts"), ("bts", "random")):
        ds = OpenBanditDataset(behavior_policy=behavior, campaign=campaign, data_path=obd_dir)
        truth = OpenBanditDataset.calc_on_policy_policy_value_estimate(
            behavior_policy=evaluation, campaign=campaign, data_path=obd_dir
        )
        errors: dict[str, list[float]] = {e: [] for e in ESTIMATORS}
        estimates: dict[str, list[float]] = {e: [] for e in ESTIMATORS}
        for i in range(runs):
            fb = ds.sample_bootstrap_bandit_feedback(sample_size=sample, random_state=i)
            if evaluation == "bts":
                policy = BernoulliTS(n_actions=ds.n_actions, len_list=ds.len_list,
                                     is_zozotown_prior=True, campaign=campaign,
                                     random_state=i)  # fmt: skip
                action_dist = policy.compute_batch_action_dist(n_sim=100_000,
                                                               n_rounds=fb["n_rounds"])  # fmt: skip
            else:
                policy = Random(n_actions=ds.n_actions, len_list=ds.len_list, random_state=i)
                action_dist = policy.compute_batch_action_dist(n_rounds=fb["n_rounds"])
            reg = RegressionModel(n_actions=ds.n_actions, len_list=ds.len_list,
                                  action_context=ds.action_context,
                                  base_model=LogisticRegression(max_iter=1000, C=100))  # fmt: skip
            q_hat = reg.fit_predict(context=fb["context"], action=fb["action"],
                                    reward=fb["reward"], position=fb["position"],
                                    pscore=fb["pscore"], n_folds=3, random_state=i)  # fmt: skip
            ope = OffPolicyEvaluation(
                bandit_feedback=fb,
                ope_estimators=[
                    DirectMethod(estimator_name="dm"),
                    InverseProbabilityWeighting(estimator_name="ipw"),
                    SelfNormalizedInverseProbabilityWeighting(estimator_name="snipw"),
                    DoublyRobust(estimator_name="dr"),
                    SelfNormalizedDoublyRobust(estimator_name="sndr"),
                    SwitchDoublyRobust(lambda_=1000.0, estimator_name="switch-dr"),
                ],
            )
            est = ope.estimate_policy_values(action_dist=action_dist,
                                             estimated_rewards_by_reg_model=q_hat)  # fmt: skip
            for name, v in est.items():
                estimates[name].append(float(v))
                errors[name].append(abs(float(v) - truth) / truth)
            print(f"  {behavior}->{evaluation} run {i + 1}/{runs}", file=sys.stderr)
        out[f"{behavior} logs -> value of {evaluation}"] = {
            "rounds_in_logs": int(ds.n_rounds),
            "behavior_policy_click_rate": r(ds.obtain_batch_bandit_feedback()["reward"].mean()),
            "true_value_of_evaluation_policy": r(truth, 5),
            "relative_error_mean": {k: r(np.mean(v)) for k, v in errors.items()},
            "relative_error_sd": {k: r(np.std(v)) for k, v in errors.items()},
            "estimate_mean": {k: r(np.mean(v), 5) for k, v in estimates.items()},
        }
    return {"campaign": campaign, "runs": runs, "bootstrap_sample": sample} | out


if __name__ == "__main__":
    args = sys.argv[1:]
    obd = Path(args[0])
    campaign = args[1] if len(args) > 1 else "men"
    runs = int(args[2]) if len(args) > 2 else 20
    sample = int(args[3]) if len(args) > 3 else 100_000
    print(json.dumps(run(obd, campaign, runs, sample), indent=1))

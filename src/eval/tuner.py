# src/tuning/tuner.py
from __future__ import annotations
import itertools, random, tempfile, yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

from src.eval.run_candidates import run_candidates_per_customer

# helpers: grid & random space
def _is_space(v: Any) -> bool:
    # grid list OR random-spec dict {min,max,type} (optionally 'choices')
    return isinstance(v, list) or (isinstance(v, dict) and ({"min","max","type"} <= set(v) or "choices" in v))

def _expand_grid(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = []
    values = []
    for k, v in params.items():
        if isinstance(v, list):
            keys.append(k); values.append(v)
        else:
            keys.append(k); values.append([v])
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

def _sample_random(params: Dict[str, Any], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    def _draw(spec):
        if isinstance(spec, dict) and "choices" in spec:
            return rng.choice(spec["choices"])
        if isinstance(spec, dict):
            lo, hi = spec["min"], spec["max"]
            if spec.get("type","float") == "int":
                return rng.randint(int(lo), int(hi))
            return rng.uniform(float(lo), float(hi))
        if isinstance(spec, list):
            return rng.choice(spec)
        return spec
    out = []
    for _ in range(n):
        out.append({k: _draw(v) for k, v in params.items()})
    return out

def _param_variants(model_cfg: Dict[str, Any], search: str, n_trials: int, seed: int) -> List[Dict[str, Any]]:
    """Return a list of concrete param dicts for this model entry.
    
    Args:
        model_cfg: model config dict from YAML (with 'params' or 'grid')
        search: "grid" or "random"
        n_trials: number of random trials (if search=="random")
        seed: random seed for reproducibility 
    
    Returns: 
        List of param dicts to try
    
    """
    rng = random.Random(seed)
    # Prefer 'grid' if present, else 'params'
    space = model_cfg.get("grid", model_cfg.get("params", {}))
    if not space:
        return [dict()]  # no params to tune
    # If any value looks like a search space, do search; else just one fixed set
    any_space = any(_is_space(v) for v in space.values())
    if not any_space:
        return [space]
    if search == "grid":
        # keep only lists (Cartesian), freeze any random specs with a few samples for tractability
        freeze = {k: (v if isinstance(v, list) else (_sample_random({k:v}, 3, rng)[0])) for k, v in space.items()}
        return _expand_grid(freeze)
    else:  # random
        return _sample_random(space, n_trials, rng)

# write/merge YAML utilities
def _single_run_yaml(base_cfg: Dict[str, Any], customer: str, model_name: str, family: str, params: Dict[str, Any]) -> str:
    cfg = {"customers": {customer: {
        "transform": base_cfg["customers"][customer].get("transform", "raw"),
        "cv": base_cfg["customers"][customer].get("cv", {}),
        "models": [{"name": model_name, "family": family, "params": params, **(
            {"use_lightgbm": True} if family == "gbm" and base_cfg["customers"][customer]
                .get("models", [{}])[0].get("use_lightgbm", False) else {}
        )}]
    }}}
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp); tmp.flush(); tmp.close()
    return tmp.name

def _pick_best(per_fold: pd.DataFrame, customer: str, model_name: str, metric: str="sMAPE") -> float:
    if per_fold.empty:
        return np.inf
    df = per_fold[(per_fold["CUSTOMER"]==customer) & (per_fold["model"]==model_name)]
    if df.empty:
        return np.inf
    # ignore _ERROR rows if any slipped through
    df = df[~df["model"].str.contains("_ERROR", na=False)]
    if df.empty:
        return np.inf
    return float(df[metric].mean())

# public API
def tune_per_customer(
    df_clean: pd.DataFrame,
    base_yaml_path: str | Path,
    *,
    search: str = "grid",            # "grid" or "random"
    n_trials: int = 40,              # used for random search
    metric: str = "sMAPE",           # lower is better
    out_yaml_path: str | Path = "configs/model_matrix_tuned.yaml",
    # pass-thru for candidate runner
    cv_defaults: Dict[str, Any] | None = None,
    features_defaults: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Try param variants per customer/model using your existing rolling-CV evaluator,
    select the best by `metric` (lower is better), and write a *new YAML* with those best params.
    Returns (per_fold_results, summary, best_cfg_dict).

    Args:
        df_clean: cleaned DataFrame with actuals
        base_yaml_path: path to base model-matrix YAML
        search: "grid" or "random"
        n_trials: number of random trials (if search=="random")
        metric: metric name to pick best (lower is better)
        out_yaml_path: path to write tuned YAML
        cv_defaults: optional dict of CV overrides to pass to runner
        features_defaults: optional dict of feature-engineering overrides to pass to runner
    
    Returns:
        per_fold_best: DataFrame of per-fold results using best params
        summary_best: DataFrame of summary results using best params
        best_cfg: Dict representing the best model-matrix YAML
        
    """
    base_cfg = yaml.safe_load(Path(base_yaml_path).read_text())

    all_rows: List[pd.Series] = []
    best_cfg: Dict[str, Any] = {"customers": {}}

    for cust, cnode in base_cfg["customers"].items():
        best_cfg["customers"][cust] = {
            "transform": cnode.get("transform","raw"),
            "cv": cnode.get("cv", {}),
            "models": []
        }
        models = cnode.get("models", [])
        if not models:
            continue

        for m in models:
            name   = m["name"]
            family = m["family"]
            variants = _param_variants(m, search=search, n_trials=n_trials, seed=1234)

            scores: List[Tuple[float, Dict[str, Any], pd.DataFrame, pd.DataFrame]] = []

            for i, params in enumerate(variants, 1):
                tmp_yaml = _single_run_yaml(base_cfg, cust, name, family, params)

                # run a single-model CV for this customer
                per_fold, summary, _, _ = run_candidates_per_customer(
                    df_clean,
                    model_matrix_path=tmp_yaml,
                    # Optional pass-through overrides
                    **(cv_defaults or {}),
                    **(features_defaults or {}),
                    save_csv=False,
                )

                score = _pick_best(per_fold, cust, name, metric=metric)
                scores.append((score, params, per_fold, summary))

            # choose best variant for this model & customer
            scores = [s for s in scores if np.isfinite(s[0])]
            if not scores:
                # keep original (no good runs); record placeholder
                best_cfg["customers"][cust]["models"].append({"name": name, "family": family, "params": m.get("params", {})})
                continue

            scores.sort(key=lambda x: x[0])  # lower metric is better
            best_score, best_params, pf_best, _ = scores[0]

            # collect rows (for report)
            all_rows.append(pd.Series({"CUSTOMER": cust, "model": name, "best_"+metric: best_score, "params": best_params}))
            # save best into final cfg
            best_cfg["customers"][cust]["models"].append({"name": name, "family": family, "params": best_params, **({"use_lightgbm": m.get("use_lightgbm", False)} if family=="gbm" else {})})

    # Write tuned YAML
    out_yaml_path = Path(out_yaml_path)
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    # Optional simple tables
    best_table = pd.DataFrame(all_rows).sort_values(["CUSTOMER","model"]).reset_index(drop=True)
    # Re-run a single consolidated CV using the best config (for one clean report)
    tmp_all = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(best_cfg, tmp_all); tmp_all.flush(); tmp_all.close()
    per_fold_best, summary_best, _, _ = run_candidates_per_customer(
        df_clean,
        model_matrix_path=tmp_all.name,
        save_csv=False,
        **(cv_defaults or {}),
        **(features_defaults or {})
    )
    return per_fold_best, summary_best, best_cfg

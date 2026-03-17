import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime


def _make_serializable(obj):
    """Convertit récursivement tout objet en type JSON-serializable."""
    if isinstance(obj, pd.DataFrame):
        if "Set" in obj.columns:
            return obj.set_index("Set").to_dict(orient="index")
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_run(results, author=None):
    """
    Sauvegarde un run ML localement.

    Crée un dossier horodaté contenant :
        - model.pkl      : le modèle entraîné
        - metrics.json   : les métriques
        - run_info.json  : infos du run (auteur, modèle, date, métriques)

    Parameters
    ----------
    results : dict
        Dict retourné par evaluate_model ou une fonction run_xxx.
        Doit contenir au minimum la clé 'model'.
    author : str, optional
        Nom de l'auteur. Si absent, utilise la variable
        d'environnement POWERFORECAST_AUTHOR, sinon 'unknown'.

    Returns
    -------
    run_dir : str
        Chemin local du dossier du run.

    Examples
    --------
    >>> run_path = save_run(results)
    >>> run_path = save_run(results, author="jean")
    """

    if "model" not in results:
        raise ValueError("results doit contenir la clé 'model'.")

    model      = results["model"]
    model_name = type(model).__name__
    run_id     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    author     = author or os.getenv("POWERFORECAST_AUTHOR", "unknown")

    # --- Dossier du run ---
    run_dir = os.path.join(
        os.getcwd(), "runs",
        f"{run_id}_{model_name}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # --- model.pkl ---
    with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # --- metrics.json ---
    metrics = _make_serializable(
        {k: v for k, v in results.items() if k != "model"}
    )
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- run_info.json ---
    run_info = {
        "run_id"    : f"{run_id}_{model_name}",
        "model"     : model_name,
        "author"    : author,
        "created_at": datetime.now().isoformat(),
        "metrics"   : metrics.get("metrics", {}),
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"✅ Run sauvegardé : {run_dir}")

    return run_dir
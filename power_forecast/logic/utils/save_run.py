"""
Utility function to save ML runs.

Centralized utility to save:
- trained models
- evaluation metrics

This function works for any model (sklearn, xgboost, tensorflow, custom models).
"""

import os
import json
import pickle
from datetime import datetime


def save_run(results):
    """
    Save a trained model and its metrics locally.

    Parameters
    ----------
    results : dict
        Dictionary returned by a model training function.
        Must contain at least:
        - model

    Optional keys (if present they will be saved):
        - train_rmse
        - test_rmse
        - train_mae
        - test_mae

    Returns
    -------
    run_dir : str
        Path of the saved run directory.
    """

    if "model" not in results:
        raise ValueError("Results dictionary must contain a 'model' key.")

    model = results["model"]

    # detect model name automatically
    model_name = type(model).__name__

    # timestamp
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # project root
    project_root = os.getcwd()

    # run directory
    run_dir = os.path.join(
        project_root,
        "runs",
        f"{run_id}_{model_name}"
    )

    os.makedirs(run_dir, exist_ok=True)

    # save model
    with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # save metrics (everything except model)
    metrics = {k: v for k, v in results.items() if k != "model"}

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Run enregistré localement dans : {run_dir}\n")

    print(
        "⚠️ Information équipe :\n"
        "Ce run est actuellement sauvegardé uniquement sur votre machine (VM ou ordinateur local).\n\n"

        "Le dossier contient :\n"
        "- model.pkl : le modèle entraîné\n"
        "- metrics.json : les métriques du run\n\n"

        "Pour partager ce run avec l'équipe :\n"
        "1️⃣ Aller dans le dossier 'runs/' du projet.\n"
        "2️⃣ Récupérer le dossier correspondant au run.\n"
        "3️⃣ Envoyer ce dossier :\n"
        "   - via Slack / Teams / Drive\n"
        "   - ou à un membre de l'équipe ayant accès au Google Cloud Storage.\n\n"

        "4️⃣ La personne ayant accès au storage peut uploader avec :\n"
        "   gsutil cp -r runs/<nom_du_run> gs://powerforecast-runs/\n"
    )

    return run_dir
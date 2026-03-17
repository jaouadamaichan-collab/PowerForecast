def run_and_log(model_fn, *args, **kwargs):
    """
    Wrapper universel autour de toute fonction d'entraînement.

    Exécute model_fn, normalise son output et logue les métriques.
    Ne calcule aucune métrique — c'est la responsabilité de model_fn.

    Parameters
    ----------
    model_fn : callable
        Fonction d'entraînement à exécuter.
        Doit retourner un dict contenant au minimum la clé 'model'.
        Exemple de retour attendu :
            {
                "model"  : <modèle entraîné>,
                "metrics": {
                    "train_mae": 4.32,
                    "val_mae"  : 9.10,
                    "test_mae" : 8.22,
                }
            }
    *args : any
        Arguments positionnels transmis à model_fn.
    **kwargs : any
        Arguments nommés transmis à model_fn.

    Returns
    -------
    results : dict
        Dict normalisé contenant 'model' et 'metrics'.

    Raises
    ------
    ValueError
        Si model_fn ne retourne pas un dict contenant 'model'.

    Examples
    --------
    >>> results = run_and_log(run_histxgb, df)
    >>> results = run_and_log(run_xgb_with_scaling, df)
    """

    # --- Exécution ---
    output = model_fn(*args, **kwargs)

    # --- Normalisation du retour ---
    if not isinstance(output, dict):
        results = {"model": output, "metrics": {}}
    else:
        results = output

    if "model" not in results:
        raise ValueError(
            f"{model_fn.__name__} doit retourner un dict "
            f"contenant la clé 'model'."
        )

    # --- Normalisation des métriques ---
    metrics = results.get("metrics", {})

    if hasattr(metrics, "to_dict"):
        if hasattr(metrics, "columns") and "Set" in metrics.columns:
            metrics = metrics.set_index("Set").to_dict(orient="index")
        else:
            metrics = metrics.to_dict()

    results["metrics"] = metrics

    # --- Log console ---
    print(f"\n{'='*50}")
    print(f"  Run    : {model_fn.__name__}")
    print(f"  Modèle : {type(results['model']).__name__}")
    if metrics:
        print(f"  Métriques :")
        _print_metrics(metrics)
    print(f"{'='*50}\n")

    return results


def _print_metrics(metrics, indent=4):
    """Affichage récursif des métriques — dict plat ou nested."""
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{' '*indent}{k}:")
            _print_metrics(v, indent + 4)
        elif isinstance(v, float):
            print(f"{' '*indent}{k:<20} : {v:.4f}")
        else:
            print(f"{' '*indent}{k:<20} : {v}")
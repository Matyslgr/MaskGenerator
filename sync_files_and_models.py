##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## sync_results_and_models
##

import os
import csv

MODELS_DIR = "Models"
RESULTS_DIR = "Results"
CROSSVAL_DIR = "crossval"

CROSSVAL_RESULTS_CSV = "crossval_results.csv"
SUMMARY_CSV = "crossval_summary.csv"

def get_model_paths_from_csv(csv_path):
    """Lit les chemins de modèle à partir du fichier CSV."""
    model_paths = set()
    rows = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_path = row.get("model_path")
            if model_path:
                model_paths.add(model_path)
                rows.append((row, model_path))
    return model_paths, rows

def synchronize_crossval_results():

    crossval_results_path = os.path.join(RESULTS_DIR, CROSSVAL_RESULTS_CSV).replace("\\", "/")

    if not os.path.exists(crossval_results_path):
        print(f"⚠️  Crossval results CSV '{crossval_results_path}' does not exist.")
        return set()

    model_paths_from_csv, rows_with_paths = get_model_paths_from_csv(crossval_results_path)

    # 1. Supprimer les lignes du CSV si le modèle n'existe pas
    valid_rows = []
    for row, path in rows_with_paths:
        if os.path.exists(path):
            valid_rows.append(row)
        else:
            print(f"⚠️  Model missing, removing CSV line: {path}")

    # Réécrire le CSV avec seulement les lignes valides
    with open(crossval_results_path, 'w', newline='') as f:
        if valid_rows:
            fieldnames = valid_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in valid_rows:
                writer.writerow(row)

    return model_paths_from_csv

def delete_unused_models(model_dir, model_paths_from_csv):
    all_models_on_disk = set()

    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".pth"):
                full_path = os.path.join(root, f).replace("\\", "/")
                all_models_on_disk.add(full_path)

    unused_models = all_models_on_disk - model_paths_from_csv
    for model_path in unused_models:
        print(f"🗑️  Model not referenced in the CSV, deleting: {model_path}")
        os.remove(model_path)

def synchronize_crossval_summary(model_paths_from_csv):

    summary_path = os.path.join(RESULTS_DIR, SUMMARY_CSV).replace("\\", "/")

    if not os.path.exists(summary_path):
        print(f"⚠️  Summary CSV '{summary_path}' does not exist.")
        return

    valid_rows = []
    with open(summary_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        model_hash = row.get("hash")
        if not model_hash:
            continue

        all_exist = True
        missing_paths = []
        for i in range(1, 6):  # Folds 1 to 5
            pattern = f"{MODELS_DIR}/{CROSSVAL_DIR}/model_{row.get('experiment_name', 'default')}_{model_hash}_fold{i}.pth"
            if pattern not in model_paths_from_csv:
                all_exist = False
                missing_paths.append(pattern)

        if all_exist:
            valid_rows.append(row)
        else:
            print(f"⚠️  Missing model(s) for hash {model_hash}, removing summary row.")
            for path in missing_paths:
                print(f"    ⛔ Missing: {path}")

    # Réécrire le CSV avec les lignes valides
    if rows:
        with open(summary_path, 'w', newline='') as f:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in valid_rows:
                writer.writerow(row)

def sync_files_and_models():
    """Fonction principale pour synchroniser results.csv et models_summary.csv."""
    if not os.path.exists(MODELS_DIR):
        print(f"⚠️ Models directory '{MODELS_DIR}' does not exist.")
        return

    model_paths_from_csv = synchronize_crossval_results()

    crossval_model_dir = os.path.join(MODELS_DIR, CROSSVAL_DIR).replace("\\", "/")
    delete_unused_models(crossval_model_dir, model_paths_from_csv)

    synchronize_crossval_summary(model_paths_from_csv)

    print("✅ Synchronization completed.")

if __name__ == "__main__":
    sync_files_and_models()

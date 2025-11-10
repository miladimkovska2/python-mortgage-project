import pandas as pd

def run_accuracy_validity_score(df_dict, rules_dict):
    total_checks_all = 0
    total_violations_all = 0

    for dataset_name, dataset in df_dict.items():
        rules = rules_dict.get(dataset_name, [])
        n_records = len(dataset)

        for rule in rules:
            col = rule["col"]
            if col in dataset.columns:
                condition = rule["condition"](dataset[col])
                total_violations_all += condition.sum()
                total_checks_all += n_records  

    score = (
        1 - (total_violations_all / total_checks_all)
        if total_checks_all > 0
        else None
    )

    return pd.DataFrame([{"OverallAccuracyValidityScore": round(score, 4)}])

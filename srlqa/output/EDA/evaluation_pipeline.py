from full_analysis_pipeline import paths, evaluate_seed_suite
p=paths(); r,s=evaluate_seed_suite(p); r.to_csv(p.tables/"model_evaluation_records.csv", index=False); s.to_csv(p.tables/"model_evaluation_summary.csv", index=False)

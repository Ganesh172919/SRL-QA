from full_analysis_pipeline import paths, inventory, load_datasets
p=paths(); inventory(p).to_csv(p.tables/"file_inventory.csv", index=False); load_datasets(p)[0].to_csv(p.tables/"dataset_overview.csv", index=False)

from src.data.feature_provider import UniFeatureProvider

provider = UniFeatureProvider("/mnt/data-surgen")
print("num labeled slides:", len(provider))

item = provider.load_slide(0)
print("slide_id:", item["slide_id"])
print("cohort:", item["cohort"])
print("case_id:", item["case_id"])
print("label:", item["label"])
print("features shape:", item["features"].shape)
print("coords shape:", item["coords"].shape)

n_pos = sum(r.label for r in provider.records)
n_neg = len(provider.records) - n_pos
print("positive:", n_pos)
print("negative:", n_neg)

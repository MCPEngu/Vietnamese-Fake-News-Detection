import pandas as pd

print("="*80)
print("INSPECTING EXTERNAL DATASETS - RAW FORMAT")
print("="*80)

# Check huynhtuan0106
print("\n1. huynhtuan0106 - Train samples:")
ht_train = pd.read_csv('data/external/huynhtuan0106/train_data.csv')
print(f"   Shape: {ht_train.shape}")
print(f"   Columns: {ht_train.columns.tolist()}")
for i in range(min(3, len(ht_train))):
    text = ht_train['content'].iloc[i]
    print(f"\n   Sample {i+1} (len={len(text)}):")
    print(f"   {repr(text[:150])}")

# Check VNFD
print("\n\n2. VNFD - File 1 (223) samples:")
vnfd = pd.read_csv('data/external/vnfd/vn_news_223_tdlfr.csv')
print(f"   Shape: {vnfd.shape}")
print(f"   Columns: {vnfd.columns.tolist()}")
for i in range(min(3, len(vnfd))):
    text = vnfd['text'].iloc[i]
    print(f"\n   Sample {i+1} (len={len(text)}):")
    print(f"   {repr(text[:150])}")

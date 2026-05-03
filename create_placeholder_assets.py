import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw

def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 1. Identity Fraud Data
print("Generating Identity Fraud sample data...")
id_path = "Identity-Fraud-Detection"
create_dirs(id_path)

train_trans = pd.DataFrame({
    'TransactionID': range(3663549, 3663649),
    'isFraud': np.random.choice([0, 1], size=100, p=[0.8, 0.2]),
    'TransactionAmt': np.random.uniform(10, 5000, 100),
    'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], 100),
    'card1': np.random.randint(1000, 20000, 100),
    'P_email_domain': np.random.choice(['gmail.com', 'yahoo.com', 'anonymous.com'], 100)
})
train_id = pd.DataFrame({
    'TransactionID': range(3663549, 3663649),
    'DeviceType': np.random.choice(['mobile', 'desktop'], 100),
    'DeviceInfo': np.random.choice(['Windows', 'iOS', 'Android'], 100)
})
train_trans.to_csv(f"{id_path}/train_transaction.csv", index=False)
train_id.to_csv(f"{id_path}/train_identity.csv", index=False)

# 2. Deepfake Sample Images
print("Generating Deepfake placeholder images...")
df_path = "Deepfake Detection with Neural Network"
create_dirs(f"{df_path}/real")
create_dirs(f"{df_path}/fake")

def generate_placeholder_img(path, color):
    img = Image.new('RGB', (128, 128), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 60), "SAMPLE MEDIA", fill=(255,255,255))
    img.save(path)

for i in range(10):
    generate_placeholder_img(f"{df_path}/real/real_{i}.jpg", (34, 139, 34)) # Green for real
    generate_placeholder_img(f"{df_path}/fake/fake_{i}.jpg", (178, 34, 34)) # Red for fake

# 3. Check Kiting Data
print("Generating Check Kiting sample data...")
kiting_path = "Check Kiting Detection using Unsupervised learning"
create_dirs(kiting_path)
kiting_df = pd.DataFrame({
    'Account_ID': np.random.randint(100, 200, 100),
    'Transaction_Amount': np.random.uniform(500, 10000, 100),
    'Balance_Before': np.random.uniform(0, 1000, 100),
    'Is_Kiting': np.random.choice([0, 1], 100, p=[0.9, 0.1])
})
kiting_df.to_csv(f"{kiting_path}/check_data.csv", index=False)

print("✅ All sample data generated successfully.")

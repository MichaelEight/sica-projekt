import wfdb
import ast
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# paths
folder_path = './ptb-xl'
ptbxl_database_path = './ptb-xl/ptbxl_database.csv'
scp_statements_path = './ptb-xl/scp_statements.csv'
filtered_path = './filtered'
training_path = './training'

# relationship with ptbxl_database.csv
df = pd.read_csv(ptbxl_database_path, index_col='ecg_id')

# filter out only 500 Hz signals
df_500 = df[df['filename_hr'].str.contains('records500', na=False)].copy()

# relationship with scp_statements.csv
df_500.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

# 8 diseases
healthy = []
front_heart_attack = []
side_heart_attack = []
bottom_heart_attack = []
back_heart_attack = []
complete_right_conduction_disorder = []
incomplete_right_conduction_disorder = []
complete_left_conduction_disorder = []

mapping = {
    'NORM': healthy,
    'AMI': front_heart_attack, 'ASMI': front_heart_attack, 'ALMI': front_heart_attack, 'INJAS': front_heart_attack, 'INJAL': front_heart_attack,
    'LMI': side_heart_attack, 'INJLA': side_heart_attack,
    'IMI': bottom_heart_attack, 'ILMI': bottom_heart_attack, 'INJIN': bottom_heart_attack,
    'PMI': back_heart_attack,
    'CRBBB': complete_right_conduction_disorder,
    'IRBBB': incomplete_right_conduction_disorder,
    'CLBBB': complete_left_conduction_disorder
}

category_map = {
    'NORM': 'healthy',
    'AMI': 'front_heart_attack', 'ASMI': 'front_heart_attack', 'ALMI': 'front_heart_attack', 'INJAS': 'front_heart_attack', 'INJAL': 'front_heart_attack',
    'LMI': 'side_heart_attack', 'INJLA': 'side_heart_attack',
    'IMI': 'bottom_heart_attack', 'ILMI': 'bottom_heart_attack', 'INJIN': 'bottom_heart_attack',
    'PMI': 'back_heart_attack',
    'CRBBB': 'complete_right_conduction_disorder',
    'IRBBB': 'incomplete_right_conduction_disorder',
    'CLBBB': 'complete_left_conduction_disorder'
}

results_for_csv = []

for i in range(df_500.shape[0]):
    sample_row = df_500.iloc[i]
    filename = sample_row['filename_hr']

    record = wfdb.rdrecord(os.path.join(folder_path, filename))

    disease_labels = sample_row['scp_codes']
    relevant_labels = {k: v for k, v in disease_labels.items() if k in mapping}

    if relevant_labels:
        best_label = max(relevant_labels, key=relevant_labels.get)
        percentage = relevant_labels[best_label]

        target_array = mapping[best_label]
        target_array.append(sample_row)

        results_for_csv.append({
            'ecg_id': sample_row.name,
            'filename_hr': filename,
            'scp_code': best_label,
            'category': category_map[best_label],
            'probability_percentage': percentage
        })

os.makedirs(filtered_path, exist_ok=True)
probabilities_df = pd.DataFrame(results_for_csv)
csv_output_path = os.path.join(filtered_path, 'disease_probabilities.csv')
probabilities_df.to_csv(csv_output_path, index=False)
print(f"Probability data successfully saved to: {csv_output_path}\n")

# 1. Load the CSV we created in the first step
csv_path = os.path.join(filtered_path, 'disease_probabilities.csv')
df_all = pd.read_csv(csv_path)

# 2. Stratified Split
# First, split into 70% train and 30% temporary (which will become val and test)
train_df, temp_df = train_test_split(
    df_all, test_size=0.30, random_state=42, stratify=df_all['category']
)

# Now, split the temporary 30% into validation (10% overall) and test (20% overall).
# 20% is exactly 2/3 of 30%, so our test_size here is 2/3.
val_df, test_df = train_test_split(
    temp_df, test_size=(2/3), random_state=42, stratify=temp_df['category']
)

# Check number of data in specific category
print("--- TOTAL DATA ---")
print(len(df_all))

print("\n--- TRAINING (70%) ---")
print(train_df['category'].value_counts())

print("\n--- VALIDATION (10%) ---")
print(val_df['category'].value_counts())

print("\n--- TEST (20%) ---")
print(test_df['category'].value_counts())

# 3. Saving files to folder based on the splits
def save_split(df, split_name):
    for _, row in df.iterrows():
        # Aim path: ./training/train/healthy/
        target_dir = os.path.join(training_path, split_name, row['category'])
        os.makedirs(target_dir, exist_ok=True)

        # Copy both .dat and .hea files
        target_array = mapping[best_label]
        target_array.append(sample_row)

# save data to specific folders
folders_and_arrays = [
    ("healthy", healthy),
    ("front_heart_attack", front_heart_attack),
    ("side_heart_attack", side_heart_attack),
    ("bottom_heart_attack", bottom_heart_attack),
    ("back_heart_attack", back_heart_attack),
    ("complete_right_conduction_disorder", complete_right_conduction_disorder),
    ("incomplete_right_conduction_disorder", incomplete_right_conduction_disorder),
    ("complete_left_conduction_disorder", complete_left_conduction_disorder)
]

for folder_name, data_array in folders_and_arrays:
    target_dir = os.path.join(filtered_path, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    for row in data_array:
        source_base = os.path.join(folder_path, row['filename_hr'])
        base_name = os.path.basename(row['filename_hr'])

        for ext in ['.dat', '.hea']:
            src = source_base + ext
            dst = os.path.join(target_dir, base_name + ext)
            if os.path.exists(src):
                shutil.copy2(src, dst)

# Save
print("\nCopying files to train, val, and test folders...")
save_split(train_df, 'train')
save_split(val_df, 'val')
save_split(test_df, 'test')
print("Ready.")
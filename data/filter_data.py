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

for i in range(df_500.shape[0]):
    sample_row = df_500.iloc[i]
    filename = sample_row['filename_hr']

    record = wfdb.rdrecord(os.path.join(folder_path, filename))

    disease_labels = sample_row['scp_codes']
    relevant_labels = {k: v for k, v in disease_labels.items() if k in mapping}

    if relevant_labels:
        best_label = max(relevant_labels, key=relevant_labels.get)
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

print("Filtering and saving complete based on highest probability.")
print("Healthy", len(healthy))
print("Front heart attack", len(front_heart_attack))
print("Side heart attack", len(side_heart_attack))
print("Bottom heart attack", len(bottom_heart_attack))
print("Back heart attack", len(back_heart_attack))
print("Complete right conduction disorder", len(complete_right_conduction_disorder))
print("Incomplete right conduction disorder", len(incomplete_right_conduction_disorder))
print("Complete left conduction disorder", len(complete_left_conduction_disorder))
print()

# divide on training, validate and test data
# categories
all_data = []
categories = {
    "healthy": healthy,
    "front_heart_attack": front_heart_attack,
    "side_heart_attack": side_heart_attack,
    "bottom_heart_attack": bottom_heart_attack,
    "back_heart_attack": back_heart_attack,
    "complete_right_conduction": complete_right_conduction_disorder,
    "incomplete_right_conduction": incomplete_right_conduction_disorder,
    "complete_left_conduction": complete_left_conduction_disorder
}

for label, data_list in categories.items():
    for row in data_list:
        # Supervise data (add label to each data)
        row_dict = row.to_dict()
        row_dict['target_label'] = label
        all_data.append(row_dict)

df_all = pd.DataFrame(all_data)

# Stratified Split
# 70% training, 30% the rest
train_df, temp_df = train_test_split(
    df_all, test_size=0.3, random_state=42, stratify=df_all['target_label']
)

# The rest - validation (10%) & test (20%)
# 0.33 from 30% is about 10% of all
test_df, val_df  = train_test_split(
    temp_df, test_size=0.3333, random_state=42, stratify=temp_df['target_label']
)

# Check number of data in specific category
print("--- TRAINING ---")
print(train_df['target_label'].value_counts())

print("\n--- VALIDATION ---")
print(val_df['target_label'].value_counts())

print("\n--- TEST ---")
print(test_df['target_label'].value_counts())


# Saving files to folder
def save_split(df, split_name):
    for _, row in df.iterrows():
        # Aim path: ./filtered/train/healthy/filename.dat
        target_dir = os.path.join(training_path, split_name, row['target_label'])
        os.makedirs(target_dir, exist_ok=True)

        source_base = os.path.join(folder_path, row['filename_hr'])
        base_name = os.path.basename(row['filename_hr'])

        for ext in ['.dat', '.hea']:
            src = source_base + ext
            dst = os.path.join(target_dir, base_name + ext)
            if os.path.exists(src):
                shutil.copy2(src, dst)


# Save
print("Copying files...")
save_split(train_df, 'train')
save_split(val_df, 'val')
save_split(test_df, 'test')
print("Ready.")
import pandas as pd
from preprocessing_utils import bt_merge_mappings , apply_bt_merge_mappings, apply_bt_gender_merge_mappings, drop_diverse_gender

"""# Files temporarily uploaded to run time"""

file_path = ""
boundary_tones = pd.read_excel(file_path, sheet_name=1)  

# Define the set of labels to discard
discard_labels = {
    "%H", "!H-", "L-%", "L-L", "-?%?", "L_", "L-(H)%", "(L)-%", "(L)-", "LHH%", "L-H*",
    "L*+H", "(L)-(L)%", "L*", "!H-L", "HL-%", "^H*", "H_", "H-HL", "H-H", "-%", "!H*",
    "L-L*", "L-L&"
}

# Discard labels
boundary_tones_cleaned = boundary_tones[~boundary_tones["1_anno_default_ns:bt"].isin(discard_labels)]

bt_replacements = [
    ("L-L%%", "L-L%"),
    ("H-L%%", "H-L%"),
    ("L-H%%", "L-H%"),
    ("L-L5", "L-L%"),
    ("H-H5", "H-H%"),
    ("H-H%%", "H-H%")
]

for to_replace, value in bt_replacements:
    boundary_tones_cleaned = boundary_tones_cleaned.replace(to_replace=to_replace, value=value)

bt_cleaned = boundary_tones_cleaned

total_bt_counts = bt_cleaned['1_anno_default_ns:bt'].value_counts()
merged_bt_counts = apply_bt_merge_mappings(total_bt_counts, bt_merge_mappings)
merged_bt_counts = merged_bt_counts.sort_values(ascending=False).reset_index(drop=False)
merged_bt_counts.columns = ['1_anno_default_ns:bt', 'count']
merged_bt_counts

################# BT Gender Counts 
number_of_bt = bt_cleaned['1_anno_default_ns:bt'].count()
print(f"Total number of Boundary Tones: {number_of_bt}")
number_of_male_speakers = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'male']['1_meta_speaker-id'].nunique()
number_of_female_speakers = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'female']['1_meta_speaker-id'].nunique()
number_of_diverse_speakers = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'diverse']['1_meta_speaker-id'].nunique()

print(f"Number of male speakers: {number_of_male_speakers}")
print(f"Number of female speakers: {number_of_female_speakers}")
print(f"Number of diverse speakers: {number_of_diverse_speakers}")
bt_cleaned['1_meta_speaker-gender'].unique()

############ drop Diverse Speaker
bt_cleaned = drop_diverse_gender(bt_cleaned)
number_of_bt = bt_cleaned['1_anno_default_ns:bt'].count()
print(f"Total number of Boundary Tones: {number_of_bt}")
number_of_male_speakers = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'male']['1_meta_speaker-id'].nunique()
number_of_female_speakers = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'female']['1_meta_speaker-id'].nunique()
number_of_diverse_speakers = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'diverse']['1_meta_speaker-id'].nunique()

print(f"Number of male speakers: {number_of_male_speakers}")
print(f"Number of female speakers: {number_of_female_speakers}")
print(f"Number of diverse speakers: {number_of_diverse_speakers}")

total_bt_counts = bt_cleaned['1_anno_default_ns:bt'].value_counts()
merged_bt_counts = apply_bt_merge_mappings(total_bt_counts, bt_merge_mappings)
merged_bt_counts = merged_bt_counts.sort_values(ascending=False).reset_index(drop=False)
merged_bt_counts.columns = ['1_anno_default_ns:bt', 'count']
merged_bt_counts

######## BT and Speaker Group Counts ##########
total_number_of_speakers = bt_cleaned['1_meta_speaker-id'].nunique()
print(f"Total number of speakers: {total_number_of_speakers}")
number_of_bilinguals = bt_cleaned[bt_cleaned['1_meta_speaker-bilingual'] == 'yes']['1_meta_speaker-id'].nunique()
print(f"Number of bilingual speakers: {number_of_bilinguals}")
number_of_monolinguals = bt_cleaned[bt_cleaned['1_meta_speaker-bilingual'] == 'no']['1_meta_speaker-id'].nunique()
print(f"Number of monolingual speakers: {number_of_monolinguals}")

total_number_of_speakers = bt_cleaned['1_meta_speaker-id'].nunique()
print(f"Total number of speakers: {total_number_of_speakers}")
number_of_bilinguals = bt_cleaned[bt_cleaned['1_meta_speaker-bilingual'] == 'yes']['1_meta_speaker-id'].nunique()
print(f"Number of bilingual speakers: {number_of_bilinguals}")
number_of_monolinguals = bt_cleaned[bt_cleaned['1_meta_speaker-bilingual'] == 'no']['1_meta_speaker-id'].nunique()
print(f"Number of monolingual speakers: {number_of_monolinguals}")

################ BT and Gender Counts ################

male_bt_count = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'male']['1_anno_default_ns:bt'].value_counts()
female_bt_count = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'female']['1_anno_default_ns:bt'].value_counts()

# Create DataFrames
male_df = pd.DataFrame(male_bt_count).reset_index()
male_df.columns = ['Boundary Tone', 'Male Count']

female_df = pd.DataFrame(female_bt_count).reset_index()
female_df.columns = ['Boundary Tone', 'Female Count']

# Merge DataFrames
gender_group_bt = pd.merge(male_df, female_df, on='Boundary Tone', how='outer').fillna(0)

# Convert counts to integers
gender_group_bt['Male Count'] = gender_group_bt['Male Count'].astype(int)
gender_group_bt['Female Count'] = gender_group_bt['Female Count'].astype(int)

gender_group_bt
merged_gender_group_bt = apply_bt_gender_merge_mappings(gender_group_bt, bt_merge_mappings)
merged_gender_group_bt = merged_gender_group_bt.sort_values(by=['Male Count', 'Female Count'], ascending=False).reset_index(drop=True)
merged_gender_group_bt

############### BT and Formality Count ###############
# Define high and low boundary tones
high_boundary_tones = ['H-H%', 'L-H%', 'H-', 'H-^H%', 'L-^H%', '!H-H%', '^H-H%']
low_boundary_tones = ['L-L%', 'H-L%', 'L-', '!H-L%', '^H-L%']


def count_boundary_tones(group, bt_list):
    return group[group['1_anno_default_ns:bt'].isin(bt_list)]['1_anno_default_ns:bt'].count()

data = []

for speaker_type, bilingual_value in [("Majority English", "yes"), ("Monolingual English", "no")]:
    for formality_level in ["formal", "informal"]:
        subset = boundary_tones_cleaned[
            (boundary_tones_cleaned['1_meta_speaker-bilingual'] == bilingual_value) &
            (boundary_tones_cleaned['1_meta_setting'] == formality_level)
        ]

        high_bt_count = count_boundary_tones(subset, high_boundary_tones)
        low_bt_count = count_boundary_tones(subset, low_boundary_tones)

        total_bt_count = high_bt_count + low_bt_count
        high_bt_percentage = (high_bt_count / total_bt_count * 100) if total_bt_count > 0 else 0
        low_bt_percentage = (low_bt_count / total_bt_count * 100) if total_bt_count > 0 else 0

        data.append({
            "Speaker Group": speaker_type,
            "Formality": formality_level,
            "High BT Count": high_bt_count,
            "Low BT Count": low_bt_count,
            "High BT Percentage": f"{high_bt_percentage:.1f}%",
            "Low BT Percentage": f"{low_bt_percentage:.1f}%"
        })

bt_counts_df = pd.DataFrame(data)
bt_counts_df

bt_cleaned.to_excel('bt_for_model.xlsx', index=False)
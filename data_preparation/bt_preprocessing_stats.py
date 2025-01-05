import pandas as pd
from preprocessing_utils import bt_merge_mappings , calculate_percentages, calculate_gender_percentages, apply_bt_merge_mappings, apply_bt_gender_merge_mappings, apply_bt_speaker_merge_mappings, drop_diverse_gender

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

# Define the replacements as a list of tuples
replacements = [
    ("L-L%%", "L-L%"),
    ("H-L%%", "H-L%"),
    ("L-H%%", "L-H%"),
    ("L-L5", "L-L%"),
    ("H-H5", "H-H%"),
    ("H-H%%", "H-H%")
]

# Apply the replacements
for to_replace, value in replacements:
    boundary_tones_cleaned = boundary_tones_cleaned.replace(to_replace=to_replace, value=value)

boundary_tones_cleaned['1_anno_default_ns:bt'].value_counts()
final_bt = pd.DataFrame(boundary_tones_cleaned)
cleaned_file_path = "/content/data_17.07.2024.xlsx"
bt_cleaned = pd.read_excel(cleaned_file_path, 1)


def total_number_of_x(df, column_name):
    return df[column_name].count()

total_number_of_x(bt_cleaned, '1_anno_default_ns:bt')

calculate_percentages(bt_cleaned, '1_anno_default_ns:bt', '1_meta_speaker-bilingual')

#  % of male and female speakers out of total number of speakers for bt_cleaned

# Calculate total number of speakers
total_speakers = bt_cleaned['1_meta_speaker-id'].nunique()

calculate_gender_percentages(bt_cleaned, '1_anno_default_ns:bt', '1_meta_speaker-gender')

boundary_tones_all_counts = bt_cleaned['1_anno_default_ns:bt'].value_counts()
apply_bt_merge_mappings(boundary_tones_all_counts, bt_merge_mappings)
total_number_of_speakers = bt_cleaned['1_meta_speaker-id'].nunique()
print(f"Total number of speakers: {total_number_of_speakers}")
number_of_bilinguals = bt_cleaned[bt_cleaned['1_meta_speaker-bilingual'] == 'yes']['1_meta_speaker-id'].nunique()
print(f"Number of bilingual speakers: {number_of_bilinguals}")
number_of_monolinguals = bt_cleaned[bt_cleaned['1_meta_speaker-bilingual'] == 'no']['1_meta_speaker-id'].nunique()
print(f"Number of monolingual speakers: {number_of_monolinguals}")
bilingual_bt_count = bt_cleaned[bt_cleaned['1_meta_speaker-bilingual'] == 'yes']['1_anno_default_ns:bt'].value_counts()
monolingual_bt_count = bt_cleaned[bt_cleaned['1_meta_speaker-bilingual'] == 'no']['1_anno_default_ns:bt'].value_counts()

# Create dataframes for bilingual and monolingual counts
bilingual_df = pd.DataFrame(bilingual_bt_count).reset_index()
bilingual_df.columns = ['Boundary Tone', 'Bilingual Count']

monolingual_df = pd.DataFrame(monolingual_bt_count).reset_index()
monolingual_df.columns = ['Boundary Tone', 'Monolingual Count']

speaker_group_bt = pd.merge(bilingual_df, monolingual_df, on='Boundary Tone', how='outer').fillna(0)

# Ensure count columns are integers
speaker_group_bt['Bilingual Count'] = speaker_group_bt['Bilingual Count'].astype(int)
speaker_group_bt['Monolingual Count'] = speaker_group_bt['Monolingual Count'].astype(int)

# Display the combined dataframe
speaker_group_bt.sort_values(by=['Bilingual Count', 'Monolingual Count'], ascending=False).reset_index(drop=True)
merged_speaker_group_bt= apply_bt_speaker_merge_mappings(speaker_group_bt, bt_merge_mappings)
merged_speaker_group_bt = merged_speaker_group_bt.sort_values(by=['Bilingual Count', 'Monolingual Count'], ascending=False).reset_index(drop=True)
merged_speaker_group_bt

# prompt: get me number of male and female speakers from bt_cleaned

# Calculate the number of male and female speakers
number_of_male_speakers = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'male']['1_meta_speaker-id'].nunique()
number_of_female_speakers = bt_cleaned[bt_cleaned['1_meta_speaker-gender'] == 'female']['1_meta_speaker-id'].nunique()

print(f"Number of male speakers: {number_of_male_speakers}")
print(f"Number of female speakers: {number_of_female_speakers}")


# prompt: counts of 2_anno_default_ns:word_bt based on male and female in 1_meta_speaker-gender

# Calculate pitch accent counts for male and female groups
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
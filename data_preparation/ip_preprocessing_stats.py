"""
Intonational Phrases Preprocessing, Absolute and Relative Counts
Calculate:

"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from preprocessing_utils import bt_merge_mappings , calculate_percentages, calculate_gender_percentages, apply_bt_merge_mappings, apply_bt_gender_merge_mappings, apply_bt_speaker_merge_mappings


# Access xslx files (temporarily uploaded to run time on Google Colab)

# Add sheet number along with file path
ip = pd.read_excel(file_path, sheet_name=2)

# Overview of the data
ip.head(50)

# discard labels set
# required to retain the phrase, just discard the label for the purpose of the count
ip_labels_to_discard = ["%H,L-","L-,L-L%", "L-", "H-", "%H", "!H-", "L-%", "L-L", "-?%?", "L_", "L-(H)%", "(L)-%","(L)-", "LHH%", "L-H*", "L*+H", "(L)-(L)%", "L*", "!H-L",  "HL-%", "^H*",  "H_", "H-HL", "H-H", "-%", "!H*", "L-L*", "L-L&"]
ip['2_anno_default_ns:word_bt'] = ip['2_anno_default_ns:word_bt'].replace(ip_labels_to_discard, np.nan)

# Define the replacements in a dictionary
ip_replacements = {
    "L-L%%": "L-L%",
    "H-L%%": "H-L%",
    "L-H%%": "L-H%",
    "L-L5": "L-L%",
    "H-H5": "H-H%",
    "H-H%%": "H-H%",
    "!H*,H-L%" : "H-L%",
    "%H,L-L%" : "L-L%",
    "L-L%,%H": "L-L%"

}

# Apply the replacements
ip.replace(to_replace=ip_replacements, inplace=True)

# Overview of the data 
ip.head(50)

final_ip = pd.DataFrame(ip)
final_ip.to_excel("/content/drive/MyDrive/final_ip.xlsx")

# Upload consolidated xslx file after preprocessing
cleaned_file_path = "/content/data_17.07.2024.xlsx"
ip_cleaned = pd.read_excel(cleaned_file_path, 2)

######################################## IP Counts #####################################

total_number_of_speakers = ip_cleaned['1_meta_speaker-id'].nunique()
print(f"Total number of speakers: {total_number_of_speakers}")
number_of_bilinguals = ip_cleaned[ip_cleaned['1_meta_speaker-bilingual'] == 'yes']['1_meta_speaker-id'].nunique()
print(f"Number of bilingual speakers: {number_of_bilinguals}")
number_of_monolinguals = ip_cleaned[ip_cleaned['1_meta_speaker-bilingual'] == 'no']['1_meta_speaker-id'].nunique()
print(f"Number of monolingual speakers: {number_of_monolinguals}")

# Calculate the number of male and female speakers
number_of_male_speakers = ip_cleaned[ip_cleaned['1_meta_speaker-gender'] == 'male']['1_meta_speaker-id'].nunique()
number_of_female_speakers = ip_cleaned[ip_cleaned['1_meta_speaker-gender'] == 'female']['1_meta_speaker-id'].nunique()

print(f"Number of male speakers: {number_of_male_speakers}")
print(f"Number of female speakers: {number_of_female_speakers}")

# number of words per IP
total_number_of_words = ip_cleaned['1_anno_default_ns:norm'].count()
print(f"Total number of words: {total_number_of_words}")

words_per_ip = total_number_of_words / ip_cleaned['2_anno_default_ns:word_bt'].count()
print(f"Words per IP: {words_per_ip}")

total_number_of_words_ip_bilingual = ip_cleaned.groupby('1_meta_speaker-bilingual')['1_anno_default_ns:norm'].count()
print(f"Total number of words by bilingual: {total_number_of_words_ip_bilingual}")

ip_total_counts = ip_cleaned['2_anno_default_ns:word_bt'].value_counts()

########################## Calculate relative frequencies  ##############################

# Calculate the value counts for bilingual and monolingual speakers
bilingual_ip_count = ip_cleaned[ip_cleaned['1_meta_speaker-bilingual'] == 'yes']['2_anno_default_ns:word_bt'].value_counts()
monolingual_ip_count = ip_cleaned[ip_cleaned['1_meta_speaker-bilingual'] == 'no']['2_anno_default_ns:word_bt'].value_counts()

# Create dataframes for bilingual and monolingual counts
bilingual_ip_df = pd.DataFrame(bilingual_ip_count).reset_index()
bilingual_ip_df.columns = ['Boundary Tone', 'Bilingual Count']

monolingual_ip_df = pd.DataFrame(monolingual_ip_count).reset_index()
monolingual_ip_df.columns = ['Boundary Tone', 'Monolingual Count']

# Merge the bilingual and monolingual dataframes
speaker_group_ip = pd.merge(bilingual_ip_df, monolingual_ip_df, on="Boundary Tone", how='outer').fillna(0)

# Ensure count columns are integers
speaker_group_ip['Bilingual Count'] = speaker_group_ip['Bilingual Count'].astype(int)
speaker_group_ip['Monolingual Count'] = speaker_group_ip['Monolingual Count'].astype(int)

# Display the combined dataframe
sorted_speaker_group_ip = speaker_group_ip.sort_values(by=['Bilingual Count', 'Monolingual Count'], ascending=False).reset_index(drop=True)
print(sorted_speaker_group_ip)


###################################################################
calculate_percentages(ip_cleaned, "2_anno_default_ns:word_bt","1_meta_speaker-bilingual" )
calculate_gender_percentages(ip_cleaned, '2_anno_default_ns:word_bt', '1_meta_speaker-gender')

ip_merge_counts = apply_bt_merge_mappings(ip_total_counts, bt_merge_mappings)

############################3 GENDER AND SPEAKER GROUP COUNTS ##############################
male_ip_count = ip_cleaned[ip_cleaned['1_meta_speaker-gender'] == 'male']['2_anno_default_ns:word_bt'].value_counts()
female_ip_count = ip_cleaned[ip_cleaned['1_meta_speaker-gender'] == 'female']['2_anno_default_ns:word_bt'].value_counts()

# Create DataFrames
male_df = pd.DataFrame(male_ip_count).reset_index()
male_df.columns = ['Boundary Tone', 'Male Count']

female_df = pd.DataFrame(female_ip_count).reset_index()
female_df.columns = ['Boundary Tone', 'Female Count']

# Merge DataFrames
gender_group_ip = pd.merge(male_df, female_df, on='Boundary Tone', how='outer').fillna(0)

# Convert counts to integers
gender_group_ip['Male Count'] = gender_group_ip['Male Count'].astype(int)
gender_group_ip['Female Count'] = gender_group_ip['Female Count'].astype(int)

gender_group_ip

merged_gender_group_ip = apply_bt_gender_merge_mappings(gender_group_ip, bt_merge_mappings)
merged_gender_group_ip = merged_gender_group_ip.sort_values(by=['Male Count', 'Female Count'], ascending=False).reset_index(drop=True)
merged_gender_group_ip


merged_speaker_group_ip= apply_bt_speaker_merge_mappings(sorted_speaker_group_ip, bt_merge_mappings)
merged_speaker_group_ip = merged_speaker_group_ip.sort_values(by=['Bilingual Count', 'Monolingual Count'], ascending=False).reset_index(drop=True)

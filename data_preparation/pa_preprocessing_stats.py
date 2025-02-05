import numpy as np
import pandas as pd
from preprocessing_utils import calculate_gender_percentages, calculate_percentages, pa_merge_mappings, apply_pa_gender_merge_mappings, apply_pa_speaker_merge_mappings, apply_pa_merge_mappings, drop_diverse_gender

"""# Files temporarily uploaded to run time"""

file_path = "/content/pitch accents redone.xlsx"
pitch_accents = pd.read_excel(file_path) #Add sheet number along with file path if needed

pitch_accents_cleaned = pitch_accents[~pitch_accents['1_meta_speaker-id'].isin(["'NULL'"])]
pitch_accents_cleaned.head(5)

"""# Discard labels set"""

pitch_accents_to_discard = ['L-L%', 'L**H', 'L-', 'H-L%', 'H-', '!H', 'L+', '*?', '*', 'L-H%', 'L*+^H*', '!H-L%', 'H-H%', 'H+L', 'L*+H*', 'L++H', 'L-H*', 'L*H*', 'L*+H%', '!H+', 'H+!H', 'L+H+']

pitch_accents_cleaned['2_anno_default_ns:word_pa'] = pitch_accents_cleaned['2_anno_default_ns:word_pa'].replace(pitch_accents_to_discard, np.nan)

# Correction labels set
pa_replacements = [
    ("H*,L-", "H*"),
    ("L*,L*", "L*"),
    ("L*,L*+H", "L*"),
    ("H*,H-L%", "H*"),
    ("L*,L-L%", "L*"),
    ("H*,L-L%", "H*"),
    ("L*,L-", "L*"),
    ("^H*,H-L%", "^H*"),
    ("L+H*, L-", "L+H*"),
    ("!H*,L-L%", "!H*"),
    ("H*,!H*", "H*"),
    ("L+^H*,L-L%", "L+^H*"),
    ("L*+H,H-", "L*+H"),
    ("H*,H-", "H*"),
    ("L*,H*", "L*"),
    ("L*H", "L+*H"),
    ("L*H+", "L*+H"),
    ("HL*", "H+L*"),
    ("H**", "H*"),
    ("H*!H", "H*+!H"),
    ("1H*", "H*"),
    ("^H*^", "^H*"),
    ("^H*,", "^H*"),
    ("L+H*,L-", "L+H*")
]

for to_replace, value in pa_replacements:
    pitch_accents_cleaned = pitch_accents_cleaned.replace(to_replace=to_replace, value=value)

pa_cleaned = pitch_accents_cleaned

total_pa_counts = pa_cleaned['2_anno_default_ns:word_pa'].value_counts()
total_number_of_words_pa = pa_cleaned['1_anno_default_ns:norm'].count()
print(f"Total number of words: {total_number_of_words_pa}")
total_number_of_words_pa_bilingual = pa_cleaned.groupby('1_meta_speaker-bilingual')['1_anno_default_ns:norm'].count()
print(f"Total number of words by bilingual: {total_number_of_words_pa_bilingual}")

merged_pa_counts = apply_pa_merge_mappings(total_pa_counts, pa_merge_mappings)
merged_pa_counts = merged_pa_counts.sort_values(ascending=False).reset_index(drop=False)
merged_pa_counts.columns = ['2_anno_default_ns:word_pa', 'count']
merged_pa_counts

number_of_words = pa_cleaned['1_anno_default_ns:norm'].count()
number_of_pa = pa_cleaned['2_anno_default_ns:word_pa'].count()
average_pa = number_of_pa / number_of_words
no_pa_percent= (number_of_words - number_of_pa) / number_of_words * 100

print(f"Total number of words: {number_of_words}")
print(f"Total number of Pitch Accents: {number_of_pa}")
print(f"Average Number of Pitch Accents per word: {average_pa}")
print(f"Percentage of words without Pitch Accents: {no_pa_percent:.2f}%")

"""# Number of male and female speakers"""

number_of_male_speakers = pa_cleaned[pa_cleaned['1_meta_speaker-gender'] == 'male']['1_meta_speaker-id'].nunique()
number_of_female_speakers = pa_cleaned[pa_cleaned['1_meta_speaker-gender'] == 'female']['1_meta_speaker-id'].nunique()
number_of_diverse_speakers = pa_cleaned[pa_cleaned['1_meta_speaker-gender'] == 'diverse']['1_meta_speaker-id'].nunique()

print(f"Number of male speakers: {number_of_male_speakers}")
print(f"Number of female speakers: {number_of_female_speakers}")
print(f"Number of diverse speakers: {number_of_diverse_speakers}")
pa_cleaned['1_meta_speaker-gender'].unique()

"""# Counts after dropping diverse speakers"""

pa_cleaned = drop_diverse_gender(pa_cleaned)

total_pa_counts = pa_cleaned['2_anno_default_ns:word_pa'].value_counts()
total_number_of_words_pa = pa_cleaned['1_anno_default_ns:norm'].count()
print(f"Total number of words: {total_number_of_words_pa}")
total_number_of_words_pa_bilingual = pa_cleaned.groupby('1_meta_speaker-bilingual')['1_anno_default_ns:norm'].count()
print(f"Total number of words by bilingual: {total_number_of_words_pa_bilingual}")

merged_pa_counts = apply_pa_merge_mappings(total_pa_counts, pa_merge_mappings)
merged_pa_counts = merged_pa_counts.sort_values(ascending=False).reset_index(drop=False)
merged_pa_counts.columns = ['2_anno_default_ns:word_pa', 'count']
merged_pa_counts

number_of_words = pa_cleaned['1_anno_default_ns:norm'].count()
number_of_pa = pa_cleaned['2_anno_default_ns:word_pa'].count()
average_pa = number_of_pa / number_of_words
no_pa_percent= (number_of_words - number_of_pa) / number_of_words * 100

print(f"Total number of words: {number_of_words}")
print(f"Total number of Pitch Accents: {number_of_pa}")
print(f"Average Number of Pitch Accents per word: {average_pa}")
print(f"Percentage of words without Pitch Accents: {no_pa_percent:.2f}%")

number_of_male_speakers = pa_cleaned[pa_cleaned['1_meta_speaker-gender'] == 'male']['1_meta_speaker-id'].nunique()
number_of_female_speakers = pa_cleaned[pa_cleaned['1_meta_speaker-gender'] == 'female']['1_meta_speaker-id'].nunique()

print(f"Number of male speakers: {number_of_male_speakers}")
print(f"Number of female speakers: {number_of_female_speakers}")

"""# PA and Speaker Group"""

# Calculate pitch accent counts for bilingual and monolingual groups
bilingual_pa_count = pa_cleaned[pa_cleaned['1_meta_speaker-bilingual'] == 'yes']['2_anno_default_ns:word_pa'].value_counts()
monolingual_pa_count = pa_cleaned[pa_cleaned['1_meta_speaker-bilingual'] == 'no']['2_anno_default_ns:word_pa'].value_counts()

bilingual_df = pd.DataFrame(bilingual_pa_count).reset_index()
bilingual_df.columns = ['Pitch Accent', 'Bilingual Count']

monolingual_df = pd.DataFrame(monolingual_pa_count).reset_index()
monolingual_df.columns = ['Pitch Accent', 'Monolingual Count']

speaker_group_pa = pd.merge(bilingual_df, monolingual_df, on='Pitch Accent', how='outer').fillna(0)

speaker_group_pa['Bilingual Count'] = speaker_group_pa['Bilingual Count'].astype(int)
speaker_group_pa['Monolingual Count'] = speaker_group_pa['Monolingual Count'].astype(int)

speaker_group_pa
merged_speaker_group_pa= apply_pa_speaker_merge_mappings(speaker_group_pa, pa_merge_mappings)
merged_speaker_group_pa = merged_speaker_group_pa.sort_values(by=['Bilingual Count', 'Monolingual Count'], ascending=False).reset_index(drop=True)
merged_speaker_group_pa

calculate_percentages(pa_cleaned, '2_anno_default_ns:word_pa', '1_meta_speaker-bilingual')

"""# PA and Gender"""

# Calculate pitch accent counts for male and female groups
male_pa_count = pa_cleaned[pa_cleaned['1_meta_speaker-gender'] == 'male']['2_anno_default_ns:word_pa'].value_counts()
female_pa_count = pa_cleaned[pa_cleaned['1_meta_speaker-gender'] == 'female']['2_anno_default_ns:word_pa'].value_counts()

# Create DataFrames
male_df = pd.DataFrame(male_pa_count).reset_index()
male_df.columns = ['Pitch Accent', 'Male Count']

female_df = pd.DataFrame(female_pa_count).reset_index()
female_df.columns = ['Pitch Accent', 'Female Count']

# Merge DataFrames
gender_group_pa = pd.merge(male_df, female_df, on='Pitch Accent', how='outer').fillna(0)

# Convert counts to integers
gender_group_pa['Male Count'] = gender_group_pa['Male Count'].astype(int)
gender_group_pa['Female Count'] = gender_group_pa['Female Count'].astype(int)

gender_group_pa

merged_gender_group_pa = apply_pa_gender_merge_mappings(gender_group_pa, pa_merge_mappings)
merged_gender_group_pa = merged_gender_group_pa.sort_values(by=['Male Count', 'Female Count'], ascending=False).reset_index(drop=True)
merged_gender_group_pa

calculate_gender_percentages(pa_cleaned,  "2_anno_default_ns:word_pa", "1_meta_speaker-gender")


pa_cleaned.to_excel('pa_for_model.xlsx', index=False)
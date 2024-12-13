"""
PA Con Preprocessing, Absolute and Relative Counts
Calculate:

"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from preprocessing_utils import pa_merge_mappings, calculate_percentages, calculate_gender_percentages, apply_pa_merge_mappings, apply_pa_speaker_merge_mappings

file_path = ""
pa_con = pd.read_excel(file_path, sheet_name=3)

# filter for 6 categories

pa_content_words = pa_con[pa_con["3_anno_default_ns:pos"].isin(["VERB", "NOUN", "ADJ", "PROPN", "NUM", "ADV"])] 

# discard labels set
pitch_accents_to_discard = ['L-L%', 'L**H', 'L-', 'H-L%', 'H-', '!H', 'L+', '*?', '*', 'L-H%', 'L*+^H*', '!H-L%', 'H-H%', 'H+L', 'L*+H*', 'L++H', 'L-H*', 'L*H*', 'L*+H%', '!H+', 'H+!H', 'L+H+']

# add NaN in place of the above label
pa_content_words['2_anno_default_ns:word_pa'] = pa_content_words['2_anno_default_ns:word_pa'].replace(pitch_accents_to_discard, np.nan)
# correction labels set
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
    pa_content_words = pa_content_words.replace(to_replace=to_replace, value=value)

pa_content_words['2_anno_default_ns:word_pa'].value_counts()

cleaned_file_path = "/content/data_17.07.2024.xlsx"
pa_con_cleaned = pd.read_excel(cleaned_file_path, 3)

####################### COUNTS ####################3
calculate_percentages(pa_con_cleaned, "2_anno_default_ns:word_pa", "1_meta_speaker-bilingual")
calculate_gender_percentages(pa_con_cleaned, '2_anno_default_ns:word_pa', '1_meta_speaker-gender')


############ MAKE A NEW COLUMN FOR WHERE THERE IS NULL VAL IN WORD_PA#################
pa_con_cleaned['new_col'] = pa_con_cleaned['2_anno_default_ns:word_pa'].notna().astype(int)

pa_con_cleaned.head(5)
pa_con_cleaned  = pa_con_cleaned[~pa_con_cleaned['1_meta_speaker-id'].isin(["'NULL'"])]
pa_con_cleaned.head(40)

#######################3
number_of_words = pa_con_cleaned['1_anno_default_ns:norm'].count()
number_of_pa = pa_con_cleaned['2_anno_default_ns:word_pa'].count()
average_pa = number_of_pa / number_of_words
no_pa_percent= (number_of_words - number_of_pa) / number_of_words * 100

print(f"Total number of words: {number_of_words}")
print(f"Total number of Pitch Accents: {number_of_pa}")
print(f"Average Number of Pitch Accents per word: {average_pa}")
print(f"Percentage of words without Pitch Accents: {no_pa_percent:.2f}%")

##########

merged_pa_con_counts = apply_pa_merge_mappings(pa_con_cleaned['2_anno_default_ns:word_pa'].value_counts(), pa_merge_mappings)
merged_pa_con_counts = merged_pa_con_counts.sort_values(ascending=False).reset_index(drop=False)
merged_pa_con_counts.columns = ['2_anno_default_ns:word_pa', 'count']
merged_pa_con_counts

# Calculate pitch accent counts for bilingual and monolingual groups
bilingual_pa_con_count = pa_con_cleaned[pa_con_cleaned['1_meta_speaker-bilingual'] == 'yes']['2_anno_default_ns:word_pa'].value_counts()
monolingual_pa_con_count = pa_con_cleaned[pa_con_cleaned['1_meta_speaker-bilingual'] == 'no']['2_anno_default_ns:word_pa'].value_counts()

bilingual_df = pd.DataFrame(bilingual_pa_con_count).reset_index()
bilingual_df.columns = ['Pitch Accent', 'Bilingual Count']

monolingual_df = pd.DataFrame(monolingual_pa_con_count).reset_index()
monolingual_df.columns = ['Pitch Accent', 'Monolingual Count']

speaker_group_pa_con = pd.merge(bilingual_df, monolingual_df, on='Pitch Accent', how='outer').fillna(0)

speaker_group_pa_con['Bilingual Count'] = speaker_group_pa_con['Bilingual Count'].astype(int)
speaker_group_pa_con['Monolingual Count'] = speaker_group_pa_con['Monolingual Count'].astype(int)


merged_speaker_group_pa_con= apply_pa_speaker_merge_mappings(speaker_group_pa_con, pa_merge_mappings)
merged_speaker_group_pa_con = merged_speaker_group_pa_con.sort_values(by=['Bilingual Count', 'Monolingual Count'], ascending=False).reset_index(drop=True)
merged_speaker_group_pa_con

#  average number of pitch accents in PA based on the 6 categories 

average_pa_per_pos = pa_con_cleaned.groupby('3_anno_default_ns:pos')['2_anno_default_ns:word_pa'].count() / pa_con_cleaned['3_anno_default_ns:pos'].value_counts()

print(average_pa_per_pos)

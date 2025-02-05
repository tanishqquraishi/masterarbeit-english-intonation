"""
Intonational Phrases Preprocessing, Absolute and Relative Counts
Calculate:

"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from preprocessing_utils import drop_diverse_gender, bt_merge_mappings , calculate_percentages, calculate_gender_percentages, apply_bt_merge_mappings, apply_bt_gender_merge_mappings, apply_bt_speaker_merge_mappings


# Access xslx files (temporarily uploaded to run time on Google Colab)

file_path = "//content/ip_redone.xlsx"
intonational_phrases = pd.read_excel(file_path)
intonational_phrases = intonational_phrases[~intonational_phrases['1_meta_speaker-id'].isin([f"'NULL'"])]
intonational_phrases.head(5)

ip = intonational_phrases

# Count the number of speakers before filtering
total_speakers = ip['1_meta_speaker-id'].nunique()
num_bilinguals = ip[ip['1_meta_speaker-bilingual'] == 'yes']['1_meta_speaker-id'].nunique()
num_monolinguals = ip[ip['1_meta_speaker-bilingual'] == 'no']['1_meta_speaker-id'].nunique()
num_males = ip[ip['1_meta_speaker-gender'] == 'male']['1_meta_speaker-id'].nunique()
num_females = ip[ip['1_meta_speaker-gender'] == 'female']['1_meta_speaker-id'].nunique()

print(f"Total speakers: {total_speakers}, Bilinguals: {num_bilinguals}, Monolinguals: {num_monolinguals}")
print(f"Male: {num_males}, Female: {num_females}")

# Speaker count not adding up, inspect data manually 
# NULL'        601 instances
ip = ip[~ip['1_meta_speaker-id'].isin([f"NULL'"])]
print(ip['1_meta_speaker-id'].value_counts())

# Drop 'diverse' gender rows and recount
ip = drop_diverse_gender(ip, '1_meta_speaker-gender')

total_speakers_after = ip['1_meta_speaker-id'].nunique()
num_males_after = ip[ip['1_meta_speaker-gender'] == 'male']['1_meta_speaker-id'].nunique()
num_females_after = ip[ip['1_meta_speaker-gender'] == 'female']['1_meta_speaker-id'].nunique()

print(f"Total speakers after filtering: {total_speakers_after}, Male: {num_males_after}, Female: {num_females_after}")

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

# Identify IP boundaries based on boundary tones
ip['ip_boundary'] = ip['2_anno_default_ns:word_bt'].notna()

# Assign unique IP IDs based on boundaries
ip_id = 1
ip_ids = []
for boundary in ip['ip_boundary']:
    ip_ids.append(ip_id)
    if boundary:
        ip_id += 1
ip['ip_id'] = ip_ids

# Count words per IP
ip_word_counts = ip.groupby('ip_id').size().reset_index(name='ip_word_count')

# Merge word counts back into the main dataset
ip = ip.merge(ip_word_counts, on='ip_id', how='left')

intonational_phrases["1_anno_default_ns:norm"].count()

# Total number of words per speaker group and formality #######
word_counts = {
    "bilingual": ip[ip['1_meta_speaker-bilingual'] == 'yes']['1_anno_default_ns:norm'].count(),
    "bilingual_formal": ip[(ip['1_meta_speaker-bilingual'] == 'yes') & (ip['1_meta_setting'] == 'formal')]['1_anno_default_ns:norm'].count(),
    "bilingual_informal": ip[(ip['1_meta_speaker-bilingual'] == 'yes') & (ip['1_meta_setting'] == 'informal')]['1_anno_default_ns:norm'].count(),
    "monolingual": ip[ip['1_meta_speaker-bilingual'] == 'no']['1_anno_default_ns:norm'].count(),
    "monolingual_formal": ip[(ip['1_meta_speaker-bilingual'] == 'no') & (ip['1_meta_setting'] == 'formal')]['1_anno_default_ns:norm'].count(),
    "monolingual_informal": ip[(ip['1_meta_speaker-bilingual'] == 'no') & (ip['1_meta_setting'] == 'informal')]['1_anno_default_ns:norm'].count()
}

for group, count in word_counts.items():
    print(f"Total number of words for {group.replace('_', ' ')}: {count}")

# IP Counts by speaker group and formality #####
ip_group_counts = {
    "bilingual": ip[ip['1_meta_speaker-bilingual'] == 'yes']['2_anno_default_ns:word_bt'].count(),
    "bilingual_formal": ip[(ip['1_meta_speaker-bilingual'] == 'yes') & (ip['1_meta_setting'] == 'formal')]['2_anno_default_ns:word_bt'].count(),
    "bilingual_informal": ip[(ip['1_meta_speaker-bilingual'] == 'yes') & (ip['1_meta_setting'] == 'informal')]['2_anno_default_ns:word_bt'].count(),
    "monolingual": ip[ip['1_meta_speaker-bilingual'] == 'no']['2_anno_default_ns:word_bt'].count(),
    "monolingual_formal": ip[(ip['1_meta_speaker-bilingual'] == 'no') & (ip['1_meta_setting'] == 'formal')]['2_anno_default_ns:word_bt'].count(),
    "monolingual_informal": ip[(ip['1_meta_speaker-bilingual'] == 'no') & (ip['1_meta_setting'] == 'informal')]['2_anno_default_ns:word_bt'].count()
}

for group, count in ip_group_counts.items():
    print(f"Total number of word boundary tones for {group.replace('_', ' ')}: {count}")
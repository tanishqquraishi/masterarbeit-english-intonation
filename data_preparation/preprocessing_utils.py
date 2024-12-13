import pandas as pd


bt_merge_mappings =  {"!H-L%": "H-L%",
                    "H-^H%" : "H-H%",
                    "^H-L%" : "H-L%",
                    "L-^H%" : "L-H%",
                    "!H-H%" : "H-H%",
                    "^H-H%": "H-H%",
}

pa_merge_mappings = {"!H*" : "H*",
                  "L+^H*" : "L+H*",
                  "^H*": "H*",
                  "L*+^H" : "L*+H",
                  "^H*+!H" : "H*+!H",
                  "^H+L*" : "H+L*",
                  "^H*+L" : "H* + L",
                  "L+!H*" : "L+H*",
                  "!H*+L" : "H* + L",
                  "L+^*H" : "L+H*",
                  "^H*" : "H*",
}

def calculate_percentages(df, word_col, bilingual_col):
    # Calculate total number of words
    total_number_of_x = df[word_col].count()

    # Calculate number of words for bilingual speakers
    number_of_x_bilingual = df[df[bilingual_col] == 'yes'][word_col].count()

    # Calculate number of words for monolingual speakers
    number_of_x_monolingual = df[df[bilingual_col] == 'no'][word_col].count()

    # Calculate percentages
    percent_of_x_bilingual = number_of_x_bilingual / total_number_of_x * 100
    percent_of_x_monolingual = number_of_x_monolingual / total_number_of_x * 100

    # Print the results
    print(f"Total number of x: {total_number_of_x}")
    print(f"Number of x for bilingual speakers: {number_of_x_bilingual}")
    print(f"Number of x for monolingual speakers: {number_of_x_monolingual}")
    print(f"Percentage of x for bilingual speakers: {percent_of_x_bilingual:.2f}%")
    print(f"Percentage of x for monolingual speakers: {percent_of_x_monolingual:.2f}%")


# Example usage:
# results = calculate_percentages(pa_cleaned, '2_anno_default_ns:word_pa', '1_meta_speaker-bilingual')


def calculate_gender_percentages(df, word_col, gender_col):
    # Calculate total number of words
    total_number_of_x = df[word_col].count()

    # Calculate number of words for male speakers
    number_of_x_male = df[df[gender_col] == 'male'][word_col].count()

    # Calculate number of words for female speakers
    number_of_x_female = df[df[gender_col] == 'female'][word_col].count()

    # Calculate relative percentages
    percent_of_x_male = number_of_x_male / total_number_of_x * 100
    percent_of_x_female = number_of_x_female / total_number_of_x * 100

    # Print the results
    print(f"Total number of x: {total_number_of_x}")
    print(f"Number of x for male speakers: {number_of_x_male}")
    print(f"Number of x for female speakers: {number_of_x_female}")
    print(f"Percentage of x for male speakers: {percent_of_x_male:.2f}%")
    print(f"Percentage of x for female speakers: {percent_of_x_female:.2f}%")

# Example usage:
# results = calculate_gender_percentages(pa_cleaned, '2_anno_default_ns:word_pa', '1_meta_speaker-gender')

####################### IP and BT ####################################################
# Function to apply merge mappings and update counts
def apply_bt_merge_mappings(bt_counts, mappings):
    bt_counts_dict = bt_counts.to_dict()
    bt_final_counts = bt_counts_dict.copy()

    for old_label, new_label in mappings.items():
        if old_label in bt_counts_dict:
            if new_label in bt_final_counts:
                bt_final_counts[new_label] += bt_counts_dict[old_label]
            else:
                bt_final_counts[new_label] = bt_counts_dict[old_label]
            # Keep the old label but set its count to 0 to retain it in the DataFrame
            bt_final_counts[old_label] = 0

    return pd.Series(bt_final_counts)

# Function to apply merge mappings and update counts based on speaker group 

def apply_bt_speaker_merge_mappings(df, mappings):
    # Iterate over the mappings
    for old_label, new_label in mappings.items():
        if old_label in df['Boundary Tone'].values:
            # Get the current counts of the old label
            old_bilingual_count = df.loc[df['Boundary Tone'] == old_label, 'Bilingual Count'].values[0]
            old_monolingual_count = df.loc[df['Boundary Tone'] == old_label, 'Monolingual Count'].values[0]

            # Check if the new label exists in the DataFrame
            if new_label in df['Boundary Tone'].values:
                # Add the counts to the existing new label
                df.loc[df['Boundary Tone'] == new_label, 'Bilingual Count'] += old_bilingual_count
                df.loc[df['Boundary Tone'] == new_label, 'Monolingual Count'] += old_monolingual_count
            else:
                # If the new label does not exist, create a new row
                new_row = pd.DataFrame([[new_label, old_bilingual_count, old_monolingual_count]],
                                       columns=['Boundary Tone', 'Bilingual Count', 'Monolingual Count'])
                df = pd.concat([df, new_row], ignore_index=True)

            # Set the counts of the old label to 0
            df.loc[df['Boundary Tone'] == old_label, ['Bilingual Count', 'Monolingual Count']] = 0

    return df


# Function to apply merge mappings and update counts based on gender 


def apply_bt_gender_merge_mappings(df, mappings):
    # Iterate over the mappings
    for old_label, new_label in mappings.items():
        if old_label in df['Boundary Tone'].values:
            # Get the current counts of the old label
            old_male_count = df.loc[df['Boundary Tone'] == old_label, 'Male Count'].values[0]
            old_female_count = df.loc[df['Boundary Tone'] == old_label, 'Female Count'].values[0]

            # Check if the new label exists in the DataFrame
            if new_label in df['Boundary Tone'].values:
                # Add the counts to the existing new label
                df.loc[df['Boundary Tone'] == new_label, 'Male Count'] += old_male_count
                df.loc[df['Boundary Tone'] == new_label, 'Female Count'] += old_female_count
            else:
                # If the new label does not exist, create a new row
                new_row = pd.DataFrame([[new_label, old_male_count, old_female_count]],
                                       columns=['Boundary Tone', 'Male Count', 'Female Count'])
                df = pd.concat([df, new_row], ignore_index=True)

            # Set the counts of the old label to 0
            df.loc[df['Boundary Tone'] == old_label, ['Male Count', 'Female Count']] = 0

    return df




############################## PA #############################

# Function to apply merge mappings and update counts
def apply_pa_merge_mappings(pa_counts, mappings):
    pa_counts_dict = pa_counts.to_dict()
    final_counts = pa_counts_dict.copy()

    for old_label, new_label in mappings.items():
        if old_label in pa_counts_dict:
            if new_label in final_counts:
                final_counts[new_label] += pa_counts_dict[old_label]
            else:
                final_counts[new_label] = pa_counts_dict[old_label]
            # Keep the old label but set its count to 0 to retain it in the DataFrame
            final_counts[old_label] = 0

    return pd.Series(final_counts)

# Function to apply merge mappings and update counts based on gender 

def apply_pa_gender_merge_mappings(df, mappings):
    # Iterate over the mappings
    for old_label, new_label in mappings.items():
        if old_label in df['Pitch Accent'].values:
            # Get the current counts of the old label
            old_male_count = df.loc[df['Pitch Accent'] == old_label, 'Male Count'].values[0]
            old_female_count = df.loc[df['Pitch Accent'] == old_label, 'Female Count'].values[0]

            # Check if the new label exists in the DataFrame
            if new_label in df['Pitch Accent'].values:
                # Add the counts to the existing new label
                df.loc[df['Pitch Accent'] == new_label, 'Male Count'] += old_male_count
                df.loc[df['Pitch Accent'] == new_label, 'Female Count'] += old_female_count
            else:
                # If the new label does not exist, create a new row
                new_row = pd.DataFrame([[new_label, old_male_count, old_female_count]],
                                       columns=['Pitch Accent', 'Male Count', 'Female Count'])
                df = pd.concat([df, new_row], ignore_index=True)

            # Set the counts of the old label to 0
            df.loc[df['Pitch Accent'] == old_label, ['Male Count', 'Female Count']] = 0

    return df

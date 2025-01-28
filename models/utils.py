import pandas as pd

"""
This module includes functions required across all or most models.

"""

def load_data(file_path):
    """
    Parameters:
    file_path (str): The path to the Excel file to be loaded.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the data from the Excel file. 
    """
    return pd.read_excel(file_path)

def create_binary_column(data, column_name, condition):
    """Create a binary column based on a condition.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the new binary column to be created.
    condition (function): A function that defines the condition for assigning binary values (eg: presence or absence of PA).

    Returns:
    pd.DataFrame: The DataFrame with the new binary column added.
    
    """
    data[column_name] = data.apply(condition, axis=1)
    return data

def rename_columns(data):
    """
    Renaming specific columns of original data extracted from ANNIS.
    Parameters:
    data (pd.DataFrame): The input DataFrame with original column names.

    Returns:
    pd.DataFrame: A DataFrame with renamed columns.

    """
    columns_dict = {
        '1_anno_default_ns:bt': 'boundary_tone',
        '2_anno_default_ns:word_pa': 'word_pa',
        '2_anno_default_ns:word_bt': 'word_bt',
        '1_meta_speaker-bilingual' : 'bilingual',
        '1_meta_setting': 'formality',
        '1_meta_speaker-gender': 'gender',
        '1_meta_speaker-id': 'speaker_id'
    }
    return data.rename(columns=columns_dict)
"""
___date__: 09 / 2024
__author__: Tanishq Quraishi

"""

import pandas as pd

"""
This module includes functions required across all or most models.

"""

def load_data(file_path):
    """Load data from an Excel file."""
    return pd.read_excel(file_path)

def create_binary_column(data, column_name, condition):
    """Create a binary column based on a condition."""
    data[column_name] = data.apply(condition, axis=1)
    return data

def rename_columns(data):
    """Renaming specific columns."""
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

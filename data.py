import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Rename the columns to more straightforward names
data = data.rename(columns={
    '1_anno_default_ns:bt': 'boundary_tone',
    '1_meta_speaker-bilingual': 'bilingual',
    '1_meta_setting': 'formality',
    '1_meta_speaker-gender': 'gender',
    '1_meta_speaker-id': 'speaker_id'
})

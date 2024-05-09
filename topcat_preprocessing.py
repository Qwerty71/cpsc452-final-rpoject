import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from missingpy import MissForest
import pickle

ROOT = "./datasets"

csv = pd.read_csv(os.path.join(ROOT, "TOPCAT Americas.csv"))
bymed = pd.read_sas(os.path.join(ROOT, "t007_allvisits_bymed.sas7bdat")) # Medication Data

cols = csv.columns.to_list()

# Separating the data into baseline and outcomes
baseline = csv[cols[:230]]
outcomes = csv[cols[230:]]

# Patients with missing BNP values and 4 more with many missing values
full = []
for i in range(len(baseline)):
    if baseline.iloc[i].BNP_VAL < 0:
        full.append(i)

full += [507, 1444, 1510, 1651]

# Columns related to KCCQ and Depression Data + Medication Data
kccq_cols = cols[3:46]
depression_cols = cols[47:59]
medication_cols = cols[146:197]

to_be_removed = kccq_cols + depression_cols + medication_cols

# Removing the KCCQ/Depression Data, and removing the missing patients
baseline = baseline.drop(full)
baseline = baseline.drop(to_be_removed, axis=1)

# Converting data from bytes to string
bymed.CODED_MED_NAME = bymed.CODED_MED_NAME.str.decode('utf-8')
bymed.MEDCAT_WHODDE = bymed.MEDCAT_WHODDE.str.decode('utf-8')
bymed.VISIT = bymed.VISIT.str.decode('utf-8')

# Retrieves list of IDs, and separates the Americas baseline dataset from the bymed dataframe
ids = baseline.ID.to_list()
topcat_americas = bymed.loc[bymed.ID.isin(ids) & (bymed.VISIT=="BASE")]

# All the medications that were put under "uncategorizable" or "not sure what this is"
ix1 = topcat_americas.loc[topcat_americas.CODED_MED_NAME == "MOM"].index # Milk of Magnesia
ix2 = topcat_americas.loc[topcat_americas.CODED_MED_NAME == "HYDRA"].index # Not sure

topcat_americas.loc[ix1, 'MEDCAT_WHODDE'] = "NON CV MED"
topcat_americas.loc[ix2, 'MEDCAT_WHODDE'] = "NON CV MED"

print(topcat_americas.loc[ix2, 'MEDCAT_WHODDE'].values)
print(topcat_americas.loc[ix2, 'CODED_MED_NAME'].values)

# Continuous values (remap=negative, deleted=delete, outlier=negative)
remap = ['cooking_salt_score', 'DM_DUR_YR', 'HEALTH_SCALE', 'DM_AGE_YR', 'HB_gdL', 'urine_val_mgg', 'BNP_VAL']
deleted = ['prior_dt1', 'elig_dt_1']
outlier = ['AST_UL', 'ALT_UL', 'TBILI_mgdL', 'ALP_UL', 'GLUCOSE_mgdL']

remap = remap + outlier

# Drop the delete columns, and remap all the other columns to NaN by threshold
baseline = baseline.drop(deleted, axis=1)

baseline[remap] = baseline[remap].mask(baseline[remap] < 0, np.nan)
baseline.urine_val_mgg = baseline.urine_val_mgg.mask(baseline.urine_val_mgg > 20000, np.nan)
baseline.ALB_gdL = baseline.ALB_gdL.mask(baseline.ALB_gdL > 30, np.nan)
baseline.WBC_kuL = baseline.WBC_kuL.mask(baseline.WBC_kuL > 100, np.nan)
baseline.gfr = baseline.gfr.mask(baseline.gfr > 250, np.nan)

# Split BNP values by type (BNP or n-terminal pro-bnp)
t002 = pd.read_sas(os.path.join(ROOT, 't002.sas7bdat'))
t002_baseline = t002.loc[t002.ID.isin(baseline.ID.to_list())]

bnp_val, pro_bnp_val = [], []

for i in range(len(t002_baseline)):
    if t002_baseline.iloc[i].BNP_TYPE == 1:
        bnp_val.append(t002_baseline.iloc[i].BNP_VAL)
        pro_bnp_val.append(0)
    else:
        bnp_val.append(0)
        pro_bnp_val.append(t002_baseline.iloc[i].BNP_VAL)

# Get rid of this column, we will add it back later
baseline = baseline.drop(['BNP_VAL'], axis=1)

# Extracting all important lists and mappings
file = open(os.path.join(ROOT, 'mappings_and_columns.dat'), 'rb')
data = pickle.load(file)
continuous_cols, categorical_cols, column_names_mapping, nonselective_beta_blockers, selective_beta_blockers = data
file.close()

continuous_cols = list(set(continuous_cols) - set(deleted))
continuous_cols.remove('BNP_VAL')

# Finding index for weight and height within the dataframe
weight_ix, height_ix = continuous_cols.index('weight'), continuous_cols.index('height')

# Arguments for Medical Transformer
combine_ace_arb = False
split_beta_blockers = True

# Custom transformers for this dataset
class ColumnAdder(BaseEstimator, TransformerMixin):
    """Adds BMI, BNP_VAL and PRO_BNP_VAL columns to the DataFrame."""
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return np.c_[X, X[:, weight_ix] / ((X[:, height_ix] / 100) ** 2), bnp_val, pro_bnp_val]
    
class MissForestImputer(BaseEstimator, TransformerMixin):
    """Imputes continuous values using MissForest."""
    def __init__(self):
        self.model = MissForest()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return self.model.fit_transform(X)
    
class AddEmptyCategory(BaseEstimator, TransformerMixin):
    """Adds category -1 for categorical values."""
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # Impute NaN, Clip all negatives to -1, clip 99 to -1, and add 1 to everything to make nonnegative
        X[np.isnan(X)] = -1
        X = X.clip(lower=-1)
        X = X.replace(99, -1)
        X += 1
        return X
    
class MedicationData(BaseEstimator, TransformerMixin):
    """Replaces current columns with binary values indicating whether the patient uses the medication.
    
    Args:
      combine_ace_arb: Decides whether to combine the ACE-I and ARB columns into one.
      split_beta_blockers: Decides whether to split the beta_blockers column into non_selective and selective categories."""

    def __init__(self, combine_ace_arb=False, split_beta_blockers=False):
        self.combine_ace_arb = combine_ace_arb
        self.split_beta_blockers = split_beta_blockers
        self.values = []
        self.mapping = column_names_mapping
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for col in self.mapping.keys():
            frequencies = []
            if col == 'ACE INHIBITORS' and self.combine_ace_arb:
              continue
            if col == 'ANGIOTENSIN RECEPTOR BLOCKERS' and self.combine_ace_arb:
              for i in ids:
                frequencies.append(0 if topcat_americas.loc[(topcat_americas.ID == i) & (topcat_americas.MEDCAT_WHODDE.isin(['ANGIOTENSIN RECEPTOR BLOCKERS', 'ACE INHIBITORS']))].empty else 1)
              self.values.append(frequencies)
              continue
            if col == 'BETA BLOCKERS' and self.split_beta_blockers:
              frequencies2 = []
              for i in ids:
                frequencies.append(0 if topcat_americas.loc[(topcat_americas.ID == i) & (topcat_americas.MEDCAT_WHODDE==col) & (topcat_americas.CODED_MED_NAME.isin(nonselective_beta_blockers))].empty else 1)
                frequencies2.append(0 if topcat_americas.loc[(topcat_americas.ID == i) & (topcat_americas.MEDCAT_WHODDE==col) & (topcat_americas.CODED_MED_NAME.isin(selective_beta_blockers))].empty else 1)
              self.values.append(frequencies)
              self.values.append(frequencies2)
              continue
            for i in ids:
                frequencies.append(0 if topcat_americas.loc[(topcat_americas.ID == i) & (topcat_americas.MEDCAT_WHODDE == col)].empty else 1)
            self.values.append(frequencies)
        return np.c_[X, np.array(self.values).T]

# Defining pipelines for continuous and categorical data.
continuous_pipeline = Pipeline([
    ('forest_imputer', MissForestImputer()),
    ('column_adder', ColumnAdder())
])

categorical_pipeline = Pipeline([
    ('empty_cat', AddEmptyCategory()),
    ('medication_data', MedicationData(combine_ace_arb=combine_ace_arb, split_beta_blockers=split_beta_blockers))
])

# Full pipeline for entire dataframe
full_pipeline = ColumnTransformer([
    ("continuous", continuous_pipeline, continuous_cols),
    ("categorical", categorical_pipeline, categorical_cols)
], remainder="passthrough")

# Output of preprocessed dataframe
baseline_processed = full_pipeline.fit_transform(baseline)

# Re-ordering the new output into original order
medication_new_cols = list(column_names_mapping.values())
previous_cols = baseline.columns.to_list()

if combine_ace_arb:
  medication_new_cols.remove('ace_inhibitors')
  medication_new_cols.remove('angiotensin_blockers')
  medication_new_cols.insert(1, "ace_arb")
if split_beta_blockers:
  medication_new_cols.remove('beta_blockers')
  index = 8
  if combine_ace_arb:
    index = 7
  medication_new_cols.insert(index, "selective_beta_blockers")
  medication_new_cols.insert(index, "nonselective_beta_blockers")

new_cols = continuous_cols + ['bmi', 'BNP_VAL', 'PRO_BNP_VAL'] + categorical_cols + medication_new_cols + ['ID']

# Output DataFrame
baseline_csv = pd.DataFrame(baseline_processed, columns=new_cols)
baseline_csv = baseline_csv[previous_cols + medication_new_cols + ['bmi', 'BNP_VAL', 'PRO_BNP_VAL']]

# Exporting to .csv format
baseline_csv.to_csv(os.path.join(ROOT, 'TOPCAT_Americas_Preprocessed.csv'), index=False)

# Creating binary file with output columns (optional)
all_cols = ['ID', continuous_cols+['bmi', 'BNP_VAL', 'PRO_BNP_VAL'], categorical_cols+medication_new_cols]
file = open(os.path.join(ROOT, 'output_cols.dat'), 'wb')
pickle.dump(all_cols, file)
file.close()
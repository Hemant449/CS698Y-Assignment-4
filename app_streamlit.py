
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Student Outcome & Fairness Demo", layout="wide")

st.title("🎓 Student Outcome Prediction & Fairness Analysis")
st.caption("A Streamlit refactor of your Colab notebook that loads a CSV, trains a model, and reports performance & fairness.\n")

# -----------------------------
# Helpers
# -----------------------------
DEFAULT_TARGET_MAPPING = {'Graduate': 2, 'Enrolled': 1, 'Dropout': 0}

PROTECTED_ATTRIBUTES_DEFAULT = [
    'Marital status',
    'Nacionality',
    'Gender',
    'Scholarship holder',
    'Displaced',
    'Debtor',
    'Age at enrollment'  # used to derive AgeBin
]

def age_to_bin(age):
    try:
        age = float(age)
    except Exception:
        return np.nan
    if age <= 20:
        return '<=20'
    elif 21 <= age <= 25:
        return '21-25'
    elif 26 <= age <= 30:
        return '26-30'
    else:
        return '>30'

@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def load_data_from_path(path: str):
    return pd.read_csv(path, sep=None, engine="python")  # auto-detect sep

def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    return missing

def compute_bias_table(df: pd.DataFrame, attr: str, target_col="Target"):
    # distribution + unfavorable rate (Dropout) table
    tbl = pd.DataFrame(index=df[attr].value_counts().index)
    tbl['count'] = df[attr].value_counts()
    tbl['representation'] = df[attr].value_counts(normalize=True)
    dropout_mask = df[target_col] == 'Dropout'
    dr = df[dropout_mask][attr].value_counts() / df[attr].value_counts()
    tbl['Dropout_rate'] = dr.fillna(0.0)
    return tbl.sort_values('count', ascending=False)

def disparate_impact_and_dpd(labels_pred: pd.Series, groups: pd.Series, favorable_label: str):
    # rate of predicted favorable outcome per group
    rates = labels_pred.groupby(groups).apply(lambda x: (x == favorable_label).mean()).sort_index()
    if (rates.max() or 0) == 0:
        di = None
        dpd = None
    else:
        base_group = rates.idxmax()
        base_rate = rates.loc[base_group]
        di = rates / base_rate
        dpd = rates - base_rate
    return rates, di, dpd

def equal_opportunity(true_labels: pd.Series, pred_labels: pd.Series, groups: pd.Series, mapping: dict, favorable_label='Graduate'):
    fv = mapping[favorable_label]
    # restrict to true favorable
    mask = true_labels == favorable_label
    tl = true_labels[mask].map(mapping)
    pl = pred_labels[mask].map(mapping)
    gp = groups[mask]
    if tl.empty:
        return None, None
    # TPR per group
    def tpr_of_group(idx):
        gmask = (gp == idx)
        if gmask.sum() == 0:
            return 0.0
        y_true = tl[gmask]
        y_pred = pl[gmask]
        if (y_true == fv).sum() == 0:
            return 0.0
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        tpr = cm[fv, fv] / (y_true == fv).sum()
        return float(tpr)
    uniq = gp.dropna().unique()
    tprs = pd.Series({g: tpr_of_group(g) for g in uniq})
    if tprs.empty:
        return None, None
    base = tprs.max()
    eo_diff = tprs - base
    return tprs, eo_diff

# -----------------------------
# Sidebar: data input
# -----------------------------
with st.sidebar:
    st.header("📥 Data")
    src = st.radio("Choose data source", ["Upload CSV", "Use local file (data.csv)"], index=0)
    if src == "Upload CSV":
        up = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    else:
        up = None
        st.info("Will attempt to read `data.csv` from the working directory.")

    st.divider()
    st.header("⚙️ Settings")
    rand_state = st.number_input("Random seed", value=42, step=1)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    favorable_label = st.selectbox("Favorable outcome label", ["Graduate", "Enrolled", "Dropout"], index=0)

    st.divider()
    st.header("🧷 Columns")
    st.caption("If your column names differ, edit them below to match your file.")
    target_col = st.text_input("Target column", value="Target")
    age_col = st.text_input("Age column used to make AgeBin", value="Age at enrollment")

# -----------------------------
# Load data
# -----------------------------
df = None
error = None
try:
    if up is not None:
        df = load_data_from_upload(up)
    else:
        if os.path.exists("data.csv"):
            df = load_data_from_path("data.csv")
        else:
            st.warning("No file uploaded and `data.csv` not found. Please upload a CSV.")
except Exception as e:
    error = str(e)

if error:
    st.error(f"Failed to load data: {error}")
    st.stop()

if df is None:
    st.stop()

st.success(f"Loaded data with shape {df.shape}")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

# Basic checks
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found. Please fix the column name in the sidebar.")
    st.stop()

# Create AgeBin if possible
if age_col in df.columns:
    df['AgeBin'] = df[age_col].apply(age_to_bin)
else:
    st.warning(f"Age column '{age_col}' not found; fairness analysis for AgeBin will be skipped.")

# Ensure target mapping
target_vals = sorted(df[target_col].dropna().unique().tolist())
target_mapping = DEFAULT_TARGET_MAPPING.copy()
# If the dataset uses different labels, try to infer a mapping to 0/1/2 in a stable order.
if set(target_vals) != set(target_mapping.keys()):
    # map in sorted order
    target_mapping = {str(label): i for i, label in enumerate(target_vals)}
inverse_target_mapping = {v: k for k, v in target_mapping.items()}

# -----------------------------
# Bias tables (descriptive)
# -----------------------------
st.subheader("📊 Bias (Descriptive) by Protected Attributes")
prot_opts = [c for c in PROTECTED_ATTRIBUTES_DEFAULT if c in df.columns]
if 'AgeBin' in df.columns:
    prot_display = prot_opts + ['AgeBin']
else:
    prot_display = prot_opts

if not prot_display:
    st.info("No protected-attribute columns found. You can still train a model from the next section.")
else:
    choice = st.multiselect("Choose attributes to summarize", prot_display, default=prot_display)
    for attr in choice:
        st.markdown(f"**Subgroup report for `{attr}`**")
        try:
            tbl = compute_bias_table(df, attr, target_col=target_col)
            st.dataframe(tbl.style.format({'representation': '{:.4f}', 'Dropout_rate': '{:.4f}'}), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute subgroup table for '{attr}': {e}")

# -----------------------------
# Train/test split & preprocessing
# -----------------------------
st.subheader("🛠️ Train Model")
with st.status("Preparing features...", expanded=False) as status:
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str).map(target_mapping)

    # Which categorical columns? Heuristic: non-numeric -> categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # If AgeBin exists, prefer it and remove raw age column from numeric_cols
    if 'AgeBin' in X.columns and age_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != age_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder='drop'
    )

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rand_state, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train_orig)
    X_test  = preprocessor.transform(X_test_orig)

    status.update(label="Features ready!", state="complete")

# -----------------------------
# Train baseline model
# -----------------------------
with st.status("Training baseline RandomForest...", expanded=False) as status:
    baseline = RandomForestClassifier(random_state=rand_state)
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    status.update(label="Baseline trained", state="complete")

acc_baseline = accuracy_score(y_test, y_pred)
st.metric("Baseline Accuracy", f"{acc_baseline:.4f}")

cr_baseline = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
st.write("**Classification report (baseline)**")
st.json(cr_baseline)

# Prepare fairness DF
y_test_cat = pd.Series(y_test, index=X_test_orig.index).map(inverse_target_mapping)
y_pred_cat = pd.Series(y_pred, index=X_test_orig.index).map(inverse_target_mapping)

# -----------------------------
# Fairness metrics (baseline)
# -----------------------------
st.subheader("⚖️ Fairness (Baseline)")
prot_for_eval = []
for c in ['Gender', 'Marital status', 'AgeBin', 'Scholarship holder', 'Displaced', 'Debtor']:
    if c in X_test_orig.columns:
        prot_for_eval.append(c)

if not prot_for_eval:
    st.info("No protected attributes available in the test set to compute fairness metrics.")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.caption("**Disparate Impact / Demographic Parity Difference** (using predicted favorable rate)")
    with col2:
        st.caption("**Equal Opportunity Difference** (TPR among true favorable)")

    for attr in prot_for_eval:
        st.markdown(f"**Attribute:** `{attr}`")
        groups = X_test_orig[attr]
        rates, di, dpd = disparate_impact_and_dpd(y_pred_cat, groups, favorable_label=favorable_label)
        if rates is not None:
            c1, c2 = st.columns(2)
            with c1:
                df_rates = pd.DataFrame({"pred_favorable_rate": rates}).sort_index()
                if di is not None:
                    df_rates["disparate_impact_vs_best"] = di
                    df_rates["dpd_vs_best"] = dpd
                st.dataframe(df_rates.style.format("{:.3f}"), use_container_width=True)
        tprs, eo = equal_opportunity(y_test_cat, y_pred_cat, groups, target_mapping, favorable_label=favorable_label)
        if tprs is not None:
            c1, c2 = st.columns(2)
            with c2:
                df_eo = pd.DataFrame({"TPR": tprs, "EO_diff_vs_best": eo}).sort_index()
                st.dataframe(df_eo.style.format("{:.3f}"), use_container_width=True)

# -----------------------------
# Bias mitigation via simple reweighing
# -----------------------------
st.subheader("🩹 Bias Mitigation (Reweighing + RandomForest)")
st.caption("Weights groups so the intersectional distribution is more uniform within each class label. Uses groups among available protected attributes.")

available_for_weighting = [c for c in ['Gender','Marital status','AgeBin','Scholarship holder','Displaced','Debtor'] if c in X_train_orig.columns]
if not available_for_weighting:
    st.info("No protected attributes available for reweighing. Skipping mitigation.")
else:
    # Build sample weights on the *original* (untransformed) training data
    train_df = X_train_orig.copy()
    train_df['Target'] = y_train.copy()

    group_cols = available_for_weighting
    # compute group counts by (groups, Target)
    grouped = train_df.groupby(group_cols + ['Target']).size().rename('n').reset_index()
    # for each Target, make target distribution over intersections uniform
    weights = pd.Series(1.0, index=train_df.index, dtype='float64')

    for tval, gdf in grouped.groupby('Target'):
        # total rows with this target
        total_t = (train_df['Target'] == tval).sum()
        if total_t == 0:
            continue
        # how many intersections exist for this target?
        k = len(gdf)
        target_prob = 1.0 / max(k, 1)
        for _, row in gdf.iterrows():
            # actual prob of this intersection given target
            mask = (train_df['Target'] == tval)
            for c in group_cols:
                mask &= (train_df[c] == row[c])
            count = mask.sum()
            if count == 0:
                continue
            p = count / total_t
            w = target_prob / max(p, 1e-6)
            weights.loc[mask] = w

    # normalize weights
    weights *= len(weights) / weights.sum()

    with st.status("Training mitigated RandomForest...", expanded=False) as status:
        mitigated = RandomForestClassifier(random_state=rand_state)
        mitigated.fit(X_train, y_train, sample_weight=weights.values)
        y_pred_m = mitigated.predict(X_test)
        status.update(label="Mitigated model trained", state="complete")

    acc_m = accuracy_score(y_test, y_pred_m)
    st.metric("Mitigated Accuracy", f"{acc_m:.4f}")

    cr_m = classification_report(y_test, y_pred_m, output_dict=True, zero_division=0)
    st.write("**Classification report (mitigated)**")
    st.json(cr_m)

    y_pred_m_cat = pd.Series(y_pred_m, index=X_test_orig.index).map(inverse_target_mapping)

    st.subheader("⚖️ Fairness (Mitigated)")
    for attr in prot_for_eval:
        st.markdown(f"**Attribute:** `{attr}`")
        groups = X_test_orig[attr]
        rates, di, dpd = disparate_impact_and_dpd(y_pred_m_cat, groups, favorable_label=favorable_label)
        if rates is not None:
            df_rates = pd.DataFrame({"pred_favorable_rate": rates}).sort_index()
            if di is not None:
                df_rates["disparate_impact_vs_best"] = di
                df_rates["dpd_vs_best"] = dpd
            st.dataframe(df_rates.style.format("{:.3f}"), use_container_width=True)
        tprs, eo = equal_opportunity(y_test_cat, y_pred_m_cat, groups, target_mapping, favorable_label=favorable_label)
        if tprs is not None:
            df_eo = pd.DataFrame({"TPR": tprs, "EO_diff_vs_best": eo}).sort_index()
            st.dataframe(df_eo.style.format("{:.3f}"), use_container_width=True)

st.info("Tip: You can download this page as a report (Menu ▸ Save > Save as PDF) or export raw tables via the three-dots menu on each dataframe.")


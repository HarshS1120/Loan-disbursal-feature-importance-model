# app.py — Disbursal Prediction Flask App
#
# Run with: python app.py
# Requires:  pip install flask pandas numpy statsmodels

import json
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ── Paths ──
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'applications_data.json')
COEFS_PATH  = os.path.join(BASE_DIR, 'coefs.json')

PREDICTORS  = ['loginCount', 'sanctionRatio', 'avgSanctionAmount']
TARGET      = 'totalDisbursedAmount'
FLOOR_PCTILE = 10   # 10th percentile floor
COMPLETENESS_THRESH = 0.5  # 50% of train mean

# ────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, 'r') as f:
        return json.load(f)

def save_data(records):
    with open(DATA_PATH, 'w') as f:
        json.dump(records, f, indent=2, default=str)

def load_coefs():
    defaults = {
        'const': -31410000,
        'loginCount': 227900,
        'sanctionRatio': 59530000,
        'avgSanctionAmount': 8.5736,
        'floor': 4454447,
        'train_means': {'loginCount': 89, 'sanctionRatio': 0.318, 'avgSanctionAmount': 1250000},
        'r2': 0.800,
        'adj_r2': 0.789,
        'mae': 5578021,
        'mape': 12.4,
        'n_train': 60
    }
    if not os.path.exists(COEFS_PATH):
        return defaults
    with open(COEFS_PATH, 'r') as f:
        saved = json.load(f)
    defaults.update(saved)
    return defaults

def save_coefs(coefs):
    with open(COEFS_PATH, 'w') as f:
        json.dump(coefs, f, indent=2)

# ────────────────────────────────────────────────
# Model fitting
# ────────────────────────────────────────────────

def fit_model(records):
    df = pd.DataFrame(records)
    for col in PREDICTORS + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['week'] = pd.to_datetime(df['week'])
    df = df.sort_values('week').reset_index(drop=True)

    df_train = df[df[TARGET] > 0].dropna(subset=PREDICTORS + [TARGET]).copy()

    if len(df_train) < 5:
        return None, 'Not enough data to fit model (need at least 5 complete rows).'

    X = sm.add_constant(df_train[PREDICTORS])
    y = df_train[TARGET]
    model = sm.OLS(y, X).fit()

    floor = float(np.percentile(y, FLOOR_PCTILE))
    fitted = model.fittedvalues.clip(lower=floor)
    mae  = float(np.abs(y - fitted).mean())
    mape = float((np.abs(y - fitted) / y).mean() * 100)

    result = {
        'const':              float(model.params['const']),
        'loginCount':         float(model.params['loginCount']),
        'sanctionRatio':      float(model.params['sanctionRatio']),
        'avgSanctionAmount':  float(model.params['avgSanctionAmount']),
        'floor':              floor,
        'train_means': {
            'loginCount':         float(df_train['loginCount'].mean()),
            'sanctionRatio':      float(df_train['sanctionRatio'].mean()),
            'avgSanctionAmount':  float(df_train['avgSanctionAmount'].mean()),
        },
        'r2':      round(model.rsquared, 3),
        'adj_r2':  round(model.rsquared_adj, 3),
        'mae':     round(mae),
        'mape':    round(mape, 1),
        'n_train': len(df_train),
        'pvalues': {
            'const':             round(float(model.pvalues['const']), 4),
            'loginCount':        round(float(model.pvalues['loginCount']), 4),
            'sanctionRatio':     round(float(model.pvalues['sanctionRatio']), 4),
            'avgSanctionAmount': round(float(model.pvalues['avgSanctionAmount']), 4),
        }
    }
    return result, None

# ────────────────────────────────────────────────
# Prediction helper
# ────────────────────────────────────────────────

def run_predict(lc, sr, asa, coefs):
    raw = (coefs['const']
           + coefs['loginCount'] * lc
           + coefs['sanctionRatio'] * sr
           + coefs['avgSanctionAmount'] * asa)
    pred = max(raw, coefs['floor'])

    means = coefs.get('train_means', {})
    thresh = COMPLETENESS_THRESH
    warnings = []
    if lc  < means.get('loginCount', 0) * thresh:
        warnings.append(f"loginCount {lc} is below 50% of train mean ({round(means['loginCount'])})")
    if sr  < means.get('sanctionRatio', 0) * thresh:
        warnings.append(f"sanctionRatio {round(sr, 3)} is below 50% of train mean ({round(means['sanctionRatio'], 3)})")
    if asa < means.get('avgSanctionAmount', 0) * thresh:
        warnings.append(f"avgSanctionAmount is below 50% of train mean")
    return pred, warnings

# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(load_data())

@app.route('/api/data', methods=['POST'])
def add_row():
    row = request.json
    records = load_data()
    # Prevent duplicate weeks
    existing_weeks = [r.get('week', '')[:10] for r in records]
    week = (row.get('week') or '')[:10]
    if week in existing_weeks:
        idx = existing_weeks.index(week)
        records[idx] = row
    else:
        records.append(row)
    save_data(records)
    return jsonify({'ok': True, 'count': len(records)})

@app.route('/api/data/import', methods=['POST'])
def import_data():
    rows = request.json
    if not isinstance(rows, list):
        return jsonify({'error': 'Expected a JSON array'}), 400
    save_data(rows)
    return jsonify({'ok': True, 'count': len(rows)})

@app.route('/api/data/<int:idx>', methods=['DELETE'])
def delete_row(idx):
    records = load_data()
    if 0 <= idx < len(records):
        records.pop(idx)
        save_data(records)
    return jsonify({'ok': True})

@app.route('/api/data/<int:idx>', methods=['PUT'])
def update_row(idx):
    records = load_data()
    if 0 <= idx < len(records):
        records[idx] = request.json
        save_data(records)
    return jsonify({'ok': True})

@app.route('/api/coefs', methods=['GET'])
def get_coefs():
    return jsonify(load_coefs())

@app.route('/api/coefs', methods=['POST'])
def update_coefs():
    coefs = load_coefs()
    coefs.update(request.json)
    save_coefs(coefs)
    return jsonify({'ok': True, 'coefs': coefs})

@app.route('/api/refit', methods=['POST'])
def refit():
    records = load_data()
    new_coefs, err = fit_model(records)
    if err:
        return jsonify({'error': err}), 400
    old_coefs = load_coefs()
    return jsonify({'ok': True, 'new_coefs': new_coefs, 'old_coefs': old_coefs})

@app.route('/api/refit/accept', methods=['POST'])
def accept_refit():
    new_coefs = request.json
    save_coefs(new_coefs)
    return jsonify({'ok': True})

@app.route('/api/predict', methods=['POST'])
def predict():
    body  = request.json
    lc    = float(body.get('loginCount', 0))
    sr    = float(body.get('sanctionRatio', 0))
    asa   = float(body.get('avgSanctionAmount', 0))
    coefs = load_coefs()
    pred, warnings = run_predict(lc, sr, asa, coefs)
    return jsonify({'prediction': round(pred), 'warnings': warnings})


if __name__ == '__main__':
    print('Starting Disbursal Prediction App...')
    print('Open http://localhost:5000 in your browser')
    app.run(debug=True, port=5000)

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import sys

# ============================================
# Parameter
# ============================================
N_BOOTSTRAP = 1000
FILEPATH = "Enter the filepath to GDP_PI_DATASET.csv that you found in the github"          # Pfad anpassen
RANDOM_SEED = 12345

# Zu testende Kombinationen
LAGS = [2, 4, 8]                         # n Jahre (Fenster und Lag gleich)
P_VALUES = [50, 66]                      # Perzentile für signifikante Änderungen
CI_LEVELS = [85,90,95,97.5,100]      # Konfidenzniveaus (in %)

# Schwellen für Prosperity Gate (aus dem Paper)
THRESHOLD_LOW = 45    # PI < 45  -> below
THRESHOLD_HIGH = 55   # PI > 55  -> above

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f.readlines()]
    header = lines[0].split(';')
    data_rows = lines[3:]
    records = []
    for row in data_rows:
        if not row or row.startswith(';;'):
            continue
        parts = row.split(';')
        if len(parts) < 2:
            continue
        year = parts[0]
        i = 1
        while i < len(parts) and i < len(header):
            country = header[i].strip()
            if country and 'EU' not in country and country not in ('', ';'):
                pi_val = parts[i].strip()
                gdp_val = parts[i+1].strip() if i+1 < len(parts) else ''
                if pi_val and gdp_val:
                    try:
                        pi = float(pi_val)
                        gdp = float(gdp_val)
                        if 0 <= pi <= 100 and gdp > 0:
                            records.append({'Country': country, 'Year': int(year),
                                            'PI': pi, 'GDP': gdp})
                    except:
                        pass
            i += 2
    df = pd.DataFrame(records).sort_values(['Country', 'Year']).reset_index(drop=True)
    print(f"Data loaded: {df['Country'].nunique()} countries, {len(df)} observations")
    return df

_RAW_DF = load_data(FILEPATH)

# ============================================
# Hilfsfunktionen
# ============================================
def compute_cumulative_changes(df, window_years):
    """Berechnet pro Land und Startjahr die prozentuale Änderung über window_years."""
    step = window_years // 2   
    records = []
    for country in df['Country'].unique():
        cdf = df[df['Country'] == country].sort_values('Year')
        if len(cdf) < step + 1:
            continue
        for i in range(len(cdf) - step):
            s = cdf.iloc[i]
            e = cdf.iloc[i + step]
            if s['PI'] > 0 and s['GDP'] > 0:
                records.append({
                    'Country': country,
                    'Start_Year': s['Year'],
                    'PI_change': (e['PI'] - s['PI']) / s['PI'] * 100,
                    'GDP_change': (e['GDP'] - s['GDP']) / s['GDP'] * 100
                })
    return pd.DataFrame(records)

def classify_and_transitions(fenster_df, lag_years, percentile):

    # Positive und negative PI-Änderungen
    pos_pi = fenster_df['PI_change'][fenster_df['PI_change'] > 0]
    neg_pi = fenster_df['PI_change'][fenster_df['PI_change'] < 0]
    pi_pos_thresh = np.percentile(pos_pi, percentile) if len(pos_pi) > 0 else np.inf
    pi_neg_thresh = np.percentile(neg_pi, 100 - percentile) if len(neg_pi) > 0 else -np.inf

    # Positive und negative GDP-Änderungen
    pos_gdp = fenster_df['GDP_change'][fenster_df['GDP_change'] > 0]
    neg_gdp = fenster_df['GDP_change'][fenster_df['GDP_change'] < 0]
    gdp_pos_thresh = np.percentile(pos_gdp, percentile) if len(pos_gdp) > 0 else np.inf
    gdp_neg_thresh = np.percentile(neg_gdp, 100 - percentile) if len(neg_gdp) > 0 else -np.inf

    def classify_pi(c):
        if c > pi_pos_thresh:
            return '+'
        elif c < pi_neg_thresh:
            return '-'
        return 'n'

    def classify_gdp(c):
        if c > gdp_pos_thresh:
            return '+'
        elif c < gdp_neg_thresh:
            return '-'
        return 'n'

    f = fenster_df.copy()
    f['PI_state'] = f['PI_change'].apply(classify_pi)
    f['GDP_state'] = f['GDP_change'].apply(classify_gdp)

    states = ['-', 'n', '+']
    pi2g = np.zeros((3, 3))
    g2pi = np.zeros((3, 3))

    for country in f['Country'].unique():
        cdf = f[f['Country'] == country].sort_values('Start_Year')
        start_map = {row['Start_Year']: row for _, row in cdf.iterrows()}
        for _, row in cdf.iterrows():
            later = row['Start_Year'] + lag_years
            if later in start_map:
                lrow = start_map[later]
                pi2g[states.index(row['PI_state']), states.index(lrow['GDP_state'])] += 1
                g2pi[states.index(row['GDP_state']), states.index(lrow['PI_state'])] += 1

    # Zeilennormalisierung
    row_sums_pi = pi2g.sum(axis=1, keepdims=True)
    pi2g = np.divide(pi2g, row_sums_pi, out=np.zeros_like(pi2g), where=row_sums_pi != 0)
    row_sums_gdp = g2pi.sum(axis=1, keepdims=True)
    g2pi = np.divide(g2pi, row_sums_gdp, out=np.zeros_like(g2pi), where=row_sums_gdp != 0)
    return pi2g, g2pi

def bootstrap_single(arguments):
    """Ein einzelner Bootstrap‑Durchlauf für einen gegebenen Datensatz."""
    fenster_df, lag_years, percentile, idx, base_seed = arguments
    rng = np.random.default_rng(base_seed + idx)
    countries = fenster_df['Country'].unique()
    n_countries = len(countries)
    sample_countries = rng.choice(countries, size=n_countries, replace=True)
    sample_fenster = pd.concat([fenster_df[fenster_df['Country'] == c] for c in sample_countries],
                               ignore_index=True)
    pi_mat, gdp_mat = classify_and_transitions(sample_fenster, lag_years, percentile)
    return pi_mat, gdp_mat

def get_bootstrap_distributions(fenster_df, lag_years, percentile, n_bootstrap):
    """Parallele Bootstrap‑Resamples, liefert zwei Arrays (n_bootstrap, 3, 3)."""
    args_iter = zip(itertools.repeat(fenster_df),
                    itertools.repeat(lag_years),
                    itertools.repeat(percentile),
                    range(n_bootstrap),
                    itertools.repeat(RANDOM_SEED))
    pi_mats = []
    gdp_mats = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(bootstrap_single, args) for args in args_iter]
        for i, future in enumerate(as_completed(futures)):
            pi_mat, gdp_mat = future.result()
            pi_mats.append(pi_mat)
            gdp_mats.append(gdp_mat)
            if (i+1) % 200 == 0:
                print(f"      Bootstrap {i+1}/{n_bootstrap} completed")
    return np.array(pi_mats), np.array(gdp_mats)

def compute_significance_from_bootstrap(pi_dist, gdp_dist, ci_level):

    if ci_level == 100:
        low_perc = 0
        high_perc = 100
    else:
        tail = (100 - ci_level) / 2
        low_perc = tail
        high_perc = 100 - tail

    # (+) Übergänge: PI->GDP (+,+) und GDP->PI (+,+)
    pi_plus = pi_dist[:, 2, 2]   # Index 2 entspricht '+'
    gdp_plus = gdp_dist[:, 2, 2]
    low_p2g_p = np.percentile(pi_plus, low_perc)
    high_p2g_p = np.percentile(pi_plus, high_perc)
    low_g2p_p = np.percentile(gdp_plus, low_perc)
    high_g2p_p = np.percentile(gdp_plus, high_perc)
    S_plus_PI = low_p2g_p - high_g2p_p      # >0 bedeutet PI->GDP dominiert
    S_plus_GDP = low_g2p_p - high_p2g_p     # >0 bedeutet GDP->PI dominiert

    # (-) Übergänge: PI->GDP (-,-) und GDP->PI (-,-)
    pi_minus = pi_dist[:, 0, 0]   # Index 0 entspricht '-'
    gdp_minus = gdp_dist[:, 0, 0]
    low_p2g_m = np.percentile(pi_minus, low_perc)
    high_p2g_m = np.percentile(pi_minus, high_perc)
    low_g2p_m = np.percentile(gdp_minus, low_perc)
    high_g2p_m = np.percentile(gdp_minus, high_perc)
    S_minus_PI = low_p2g_m - high_g2p_m
    S_minus_GDP = low_g2p_m - high_p2g_m

    # Entscheidungen: signifikant wenn S > 0
    sig_PIplus = (S_plus_PI > 0)
    sig_PIminus = (S_minus_PI > 0)
    sig_GDPplus = (S_plus_GDP > 0)
    sig_GDPminus = (S_minus_GDP > 0)

    return sig_PIplus, sig_PIminus, sig_GDPplus, sig_GDPminus

def get_dataset(df, subset_type):
    """Liefert den gefilterten DataFrame für 'all', 'below' oder 'above'."""
    if subset_type == 'all':
        return df
    elif subset_type == 'below':
        return df[df['PI'] < THRESHOLD_LOW].copy()
    elif subset_type == 'above':
        return df[df['PI'] > THRESHOLD_HIGH].copy()
    else:
        raise ValueError("subset_type must be 'all', 'below', or 'above'")

def run_for_combination(n, p, ci, subset_type):
    """Führt die gesamte Analyse für eine (n, p, ci, subset) durch."""
    print(f"  Running n={n}, p={p}, ci={ci}, subset={subset_type}...")
    df_sub = get_dataset(_RAW_DF, subset_type)
    if len(df_sub) < 50 or df_sub['Country'].nunique() < 5:
        print(f"    -> Insufficient data, skipping.")
        return (None, None, None, None)

    # Fenster und Lag sind gleich n
    fenster_df = compute_cumulative_changes(df_sub, window_years=n)
    if len(fenster_df) < 20:
        print(f"    -> Too few change windows ({len(fenster_df)}), skipping.")
        return (None, None, None, None)

    pi_dist, gdp_dist = get_bootstrap_distributions(fenster_df, lag_years=n,
                                                    percentile=p, n_bootstrap=N_BOOTSTRAP)
    sig = compute_significance_from_bootstrap(pi_dist, gdp_dist, ci)
    return sig   # (PIplus, PIminus, GDPplus, GDPminus)

# ============================================
# Hauptschleife: Tabellen generieren
# ============================================
def main():

    results = {}
    for p in P_VALUES:
        results[p] = {}
        for ci in CI_LEVELS:
            results[p][ci] = {}
            for subset in ['all', 'below', 'above']:
                results[p][ci][subset] = {}
                for lag in LAGS:
                    results[p][ci][subset][lag] = None

    total = len(P_VALUES) * len(CI_LEVELS) * 3 * len(LAGS)
    print(f"Total combinations to process: {total}")
    counter = 0
    for p in P_VALUES:
        for ci in CI_LEVELS:
            for subset in ['all', 'below', 'above']:
                for lag in LAGS:
                    counter += 1
                    print(f"\n[{counter}/{total}] p={p}, ci={ci}, subset={subset}, lag={lag}")
                    sig = run_for_combination(lag, p, ci, subset)
                    results[p][ci][subset][lag] = sig

    # ============================================
    # Tabellen ausgeben
    # ============================================

    for p in P_VALUES:
        print(f"\n\n{'='*80}")
        print(f"RESULTS FOR p = {p}")
        print('='*80)
        for ci in CI_LEVELS:
            print(f"\n--- Confidence Level: {ci}% ---")
            # Tabelle für dieses ci
            # Kopfzeile: Entire Data Set, Below Gate, Above Gate
            # Unterkopf: 2y,4y,8y für jeden
            cols = []
            for subset in ['all', 'below', 'above']:
                for lag in LAGS:
                    cols.append(f"{subset}_{lag}y")
            # Zeilen: PI+→GDP+, PI-→GDP-, GDP+→PI+, GDP-→PI-
            rows = ['PI+→GDP+', 'PI-→GDP-', 'GDP+→PI+', 'GDP-→PI-']
            data = []
            for row in rows:
                row_data = []
                for subset in ['all', 'below', 'above']:
                    for lag in LAGS:
                        sig_tuple = results[p][ci][subset][lag]
                        if sig_tuple is None:
                            val = 'No'
                        else:
                            if row == 'PI+→GDP+':
                                val = 'Yes' if sig_tuple[0] else 'No'
                            elif row == 'PI-→GDP-':
                                val = 'Yes' if sig_tuple[1] else 'No'
                            elif row == 'GDP+→PI+':
                                val = 'Yes' if sig_tuple[2] else 'No'
                            elif row == 'GDP-→PI-':
                                val = 'Yes' if sig_tuple[3] else 'No'
                            else:
                                val = ''
                        row_data.append(val)
                data.append(row_data)

            df_table = pd.DataFrame(data, index=rows, columns=cols)
            print(df_table.to_string())
            print("\n" + "-"*60)

if __name__ == "__main__":
    main()
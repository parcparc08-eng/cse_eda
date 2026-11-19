"""Explore stage distance and climbing across the three cycling Grand Tours."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'grand-tours-stages.jsonl'
OUTPUT = ROOT / 'output'
METRICS = ['Distance', 'Elevation']
RACES = {'TdF': 'Tour de France', 'GdI': "Giro d'Italia", 'Vuelta': 'Vuelta'}
COLORS = {'TdF': '#206695', 'GdI': '#9a3d69', 'Vuelta': '#9e4a18'}
MARKERS = {'TdF': 'o', 'GdI': 's', 'Vuelta': '^'}
TERRAIN = ['Flat', 'Hilly', 'Mountainous']


def clean_stages(data):
    """Remove exact duplicates and rows missing an analysis field."""
    stages = data.drop_duplicates().dropna(
        subset=['id', 'Race', 'Year', 'RouteType'] + METRICS
    )
    stages = stages.loc[stages['RouteType'].str.strip().ne('')].copy()
    if stages['id'].duplicated().any():
        raise ValueError('Conflicting records share a stage ID.')
    return stages


def summarize_editions(stages):
    """Return one row per race and year with count and stage means."""
    return stages.groupby(['Race', 'Year'], as_index=False).agg(
        Stages=('id', 'size'),
        Distance=('Distance', 'mean'),
        Elevation=('Elevation', 'mean'),
    )


def fit_distance_model(stages):
    """Estimate a common distance slope with race and clustered errors."""
    model_data = stages.assign(
        Distance10=stages['Distance'] / 10,
        Edition=stages['Race'] + '-' + stages['Year'].astype(str),
    )
    return smf.ols(
        'Elevation ~ Distance10 + C(Race)', data=model_data
    ).fit(
        cov_type='cluster',
        cov_kwds={'groups': model_data['Edition']},
        use_t=True,
    )


def summarize_trends(editions):
    """Fit a descriptive annual slope for each race and metric."""
    estimates = []
    for race, group in editions.groupby('Race'):
        for metric in METRICS:
            fit = smf.ols(f'{metric} ~ Year', data=group).fit()
            estimates.append({
                'Race': race, 'Metric': metric,
                'Editions': len(group), 'SlopePerYear': fit.params['Year'],
            })
    return pd.DataFrame(estimates)


def summarize_race_slopes(stages):
    """Describe how the distance slope differs across races."""
    estimates = []
    for race, group in stages.groupby('Race'):
        data = group.assign(Distance10=group['Distance'] / 10)
        fit = smf.ols('Elevation ~ Distance10', data=data).fit()
        estimates.append({
            'Race': race, 'Stages': len(group),
            'SlopeMetersPer10km': fit.params['Distance10'],
        })
    return pd.DataFrame(estimates)


def save_figure(fig, filename):
    """Write a figure to the output folder and release it."""
    fig.savefig(OUTPUT / filename, dpi=220)
    plt.close(fig)


def plot_distance_and_elevation(stages):
    """Make the RQ1 scatterplot and identify each race by color and shape."""
    fig, ax = plt.subplots(figsize=(7.0, 3.15), layout='constrained')
    sns.scatterplot(
        data=stages, x='Distance', y='Elevation', hue='Race', style='Race',
        hue_order=list(RACES), style_order=list(RACES), palette=COLORS,
        markers=MARKERS, alpha=0.64, s=26, linewidth=0, ax=ax,
    )
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, list(RACES.values()), loc='upper left', frameon=False)
    ax.set(
        title='Stage distance and climbing by race',
        xlabel='Stage distance (km)', ylabel='Elevation gain (m)',
    )
    save_figure(fig, 'rq1_distance_elevation.png')


def plot_terrain(stages):
    """Compare the stage distance and climbing distributions by terrain."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), layout='constrained')
    for ax, metric, label in zip(
        axes, METRICS, ['Stage distance (km)', 'Elevation gain (m)']
    ):
        sns.boxplot(
            data=stages, x='RouteType', y=metric, order=TERRAIN,
            color='#aac6dc', width=0.58, fliersize=2.5, ax=ax,
        )
        ax.set(xlabel='', ylabel=label)
    fig.suptitle('Stage distance and climbing by terrain', fontweight='bold')
    save_figure(fig, 'rq2_terrain.png')


def plot_editions(editions):
    """Plot average stage distance and climbing for each edition."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), layout='constrained')
    styles = {'TdF': '-', 'GdI': '--', 'Vuelta': '-.'}
    for ax, metric, label in zip(
        axes, METRICS, ['Mean stage distance (km)', 'Mean elevation gain (m)']
    ):
        for race, group in editions.groupby('Race'):
            ax.plot(
                group['Year'], group[metric], color=COLORS[race],
                marker=MARKERS[race], linestyle=styles[race],
                markersize=3.8, label=RACES[race],
            )
        ax.set(xlabel='Year', ylabel=label, xticks=[2018, 2020, 2022, 2024])
    axes[0].legend(loc='upper right', frameon=False, fontsize=8.5)
    fig.suptitle('Stage averages by race edition', fontweight='bold')
    save_figure(fig, 'rq3_editions.png')


def plot_diagnostics(model):
    """Check for patterns in the residuals and inspect their distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), layout='constrained')
    sns.scatterplot(
        x=model.fittedvalues, y=model.resid, alpha=0.6, s=19,
        color='#206695', linewidth=0, ax=axes[0],
    )
    axes[0].axhline(0, color='#373737', linestyle='--', linewidth=1)
    axes[0].set(
        title='Residuals vs. fitted values',
        xlabel='Fitted elevation gain (m)',
        ylabel='Residual (m)',
    )
    sm.qqplot(model.resid, line='s', ax=axes[1], marker='.', markersize=4)
    axes[1].set(title='Normal Q-Q plot', ylabel='Residual quantiles (m)')
    fig.suptitle('Distance model diagnostics', fontweight='bold')
    save_figure(fig, 'rq1_diagnostics.png')


def save_tables(raw, stages, editions):
    """Save tables for the data audit and the descriptive analyses."""
    blanks = raw.select_dtypes(include='object').apply(
        lambda column: column.str.strip().eq('').sum()
    )
    audit = raw.isna().sum().rename('NullCells').to_frame()
    audit['BlankStrings'] = blanks.reindex(raw.columns, fill_value=0)
    audit.to_csv(OUTPUT / 'missing_values.csv')
    stats = stages[METRICS].describe().drop(index='count')
    stats.to_csv(OUTPUT / 'numeric_summary.csv')
    for column in ['Race', 'Year', 'RouteType']:
        counts = stages[column].value_counts().sort_index()
        counts.to_csv(OUTPUT / f'counts_{column.lower()}.csv')
    terrain = stages.groupby('RouteType')[METRICS].agg(['size', 'median'])
    terrain.to_csv(OUTPUT / 'terrain_summary.csv')
    editions.to_csv(OUTPUT / 'edition_summary.csv', index=False)
    summarize_trends(editions).to_csv(
        OUTPUT / 'annual_trends.csv', index=False
    )
    summarize_race_slopes(stages).to_csv(
        OUTPUT / 'race_distance_slopes.csv', index=False
    )


def summarize_distance_fits(stages, main_fit):
    """Compare the main result to a fit for stages at least 50 km."""
    short_cutoff = stages.loc[stages['Distance'] >= 50]
    results = []
    for label, fit in [
        ('All complete stages', main_fit),
        ('Stages at least 50 km', fit_distance_model(short_cutoff)),
    ]:
        lower, upper = fit.conf_int().loc['Distance10']
        results.append({
            'Sample': label, 'Stages': int(fit.nobs),
            'SlopeMetersPer10km': fit.params['Distance10'],
            'Lower95': lower, 'Upper95': upper,
            'PValue': fit.pvalues['Distance10'],
        })
    return pd.DataFrame(results)


def main():
    """Run the full analysis and save its tables and figures."""
    matplotlib.use('Agg')
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.16)
    OUTPUT.mkdir(exist_ok=True)
    raw = pd.read_json(DATA_PATH, lines=True)
    stages = clean_stages(raw)
    editions = summarize_editions(stages)
    model = fit_distance_model(stages)
    save_tables(raw, stages, editions)
    plot_distance_and_elevation(stages)
    plot_terrain(stages)
    plot_editions(editions)
    plot_diagnostics(model)
    distance_fits = summarize_distance_fits(stages, model)
    distance_fits.to_csv(OUTPUT / 'distance_model.csv', index=False)
    print(f'{raw.shape[0]} rows and {raw.shape[1]} columns before cleaning')
    print(f'{raw.drop_duplicates().shape[0]} rows after de-duplication')
    print(f'{len(stages)} records across {len(editions)} editions analyzed')
    print(distance_fits.to_string(index=False))


if __name__ == '__main__':
    main()

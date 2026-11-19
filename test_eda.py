"""Check the stage cleaning, edition summaries, and fitted distance slope."""

import pandas as pd

from eda import clean_stages, fit_distance_model, summarize_editions


def test_clean_stages():
    """Keep distinct valid rows and discard repeated or incomplete rows."""
    rows = [
        ['A', 'TdF', 2018, 'Flat', 100, 200],
        ['A', 'TdF', 2018, 'Flat', 100, 200],
        ['B', 'TdF', 2018, 'Hilly', 130, 600],
        ['C', 'Vuelta', 2020, '', 120, 350],
        ['D', 'Vuelta', 2020, 'Flat', None, 300],
    ]
    columns = ['id', 'Race', 'Year', 'RouteType', 'Distance', 'Elevation']
    cleaned = clean_stages(pd.DataFrame(rows, columns=columns))
    assert cleaned['id'].tolist() == ['A', 'B']


def test_summarize_editions():
    """Check that years and races are grouped separately."""
    data = pd.DataFrame({
        'id': ['A', 'B', 'C', 'D'],
        'Race': ['TdF', 'TdF', 'TdF', 'Vuelta'],
        'Year': [2018, 2018, 2019, 2018],
        'Distance': [100, 140, 150, 160],
        'Elevation': [200, 300, 400, 500],
    })
    summary = summarize_editions(data).set_index(['Race', 'Year'])
    assert summary.loc[('TdF', 2018)].tolist() == [2, 120, 250]
    assert summary.loc[('TdF', 2019)].tolist() == [1, 150, 400]
    assert summary.loc[('Vuelta', 2018)].tolist() == [1, 160, 500]


def test_distance_units():
    """Recover a known effect of 40 meters for every 10 kilometers."""
    data = pd.DataFrame({
        'Race': ['TdF'] * 4 + ['Vuelta'] * 4,
        'Year': [2018, 2018, 2019, 2019] * 2,
        'Distance': [100, 130, 140, 150] * 2,
        'Elevation': [500, 620, 660, 700, 800, 920, 960, 1000],
    })
    result = fit_distance_model(data)
    assert abs(result.params['Distance10'] - 40) < 1e-8


def main():
    """Run all tests from the command line."""
    test_clean_stages()
    test_summarize_editions()
    test_distance_units()
    print('All three EDA tests passed.')


if __name__ == '__main__':
    main()

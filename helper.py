import numpy as np

def fetch_medal_tally(df, year, country):
    # keep unique medal records
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0

    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df.copy()
    elif year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    elif year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    else:  # year != 'Overall' and country != 'Overall'
        temp_df = medal_df[(medal_df['Year'] == int(year)) & (medal_df['region'] == country)]

    if flag == 1:
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()

    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    # ensure integer dtype (avoid float due to NaNs)
    x['Gold'] = x['Gold'].fillna(0).astype(int)
    x['Silver'] = x['Silver'].fillna(0).astype(int)
    x['Bronze'] = x['Bronze'].fillna(0).astype(int)
    x['total'] = x['total'].fillna(0).astype(int)

    return x


def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years, country


def data_over_time(df, col):
    """
    Returns a DataFrame with columns:
      - 'Edition'  (ascending Year)
      - <col>       (count of unique (Year, col) combinations)

    This produces a consistent schema so calling code can use:
        px.line(nations_over_time, x="Edition", y="region")
    """
    # keep unique Year-col combinations
    df_unique = df.drop_duplicates(subset=['Year', col])
    # count unique occurrences per Year
    counts = df_unique.groupby('Year').size().reset_index(name=col)
    # sort by Year and rename Year -> Edition to match existing app usage
    counts = counts.sort_values('Year').reset_index(drop=True)
    counts.rename(columns={'Year': 'Edition'}, inplace=True)
    return counts


def most_successful(df, sport):
    # consider only medal-winning records
    temp_df = df.dropna(subset=['Medal']).copy()

    # if a specific sport is selected, filter
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    # get name counts (number of medals per athlete), with explicit column names
    counts = temp_df['Name'].value_counts().reset_index(name='Medals')
    counts.columns = ['Name', 'Medals']

    # keep top 15 athletes
    top = counts.head(15)

    # merge with original df to fetch Sport and region info (left join on Name)
    # use a subset of df columns to avoid duplicating many rows
    info = df[['Name', 'Sport', 'region']].drop_duplicates(subset=['Name'])

    result = top.merge(info, on='Name', how='left')

    # ensure ordering by Medals desc (value_counts() already gives this, but safe)
    result = result.sort_values('Medals', ascending=False).reset_index(drop=True)

    return result



def yearwise_medal_tally(df, country):
    temp_df = df.dropna(subset=['Medal']).copy()
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()
    final_df.rename(columns={'Medal': 'Medal'}, inplace=True)
    return final_df


def country_event_heatmap(df, country):
    temp_df = df.dropna(subset=['Medal']).copy()
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt


def most_successful_countrywise(df, country):
    """
    Return top 10 athletes for a given country with columns:
      - Name
      - Medals
      - Sport

    This is robust across pandas versions by using explicit column names.
    """
    # only consider medal-winning records and make a copy
    temp_df = df.dropna(subset=['Medal']).copy()

    # filter for the requested country
    temp_df = temp_df[temp_df['region'] == country]

    # count medals per athlete with explicit column names
    counts = temp_df['Name'].value_counts().reset_index(name='Medals')
    counts.columns = ['Name', 'Medals']

    # keep top 10
    top10 = counts.head(10)

    # get one-row-per-athlete info (choose first encountered sport — can be refined)
    info = df[['Name', 'Sport']].drop_duplicates(subset=['Name'])

    # merge to attach sport to each athlete
    result = top10.merge(info, on='Name', how='left')

    # ensure ordering by Medals desc
    result = result.sort_values('Medals', ascending=False).reset_index(drop=True)

    return result



def weight_v_height(df, sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region']).copy()
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df


def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region']).copy()

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)
    final.fillna(0, inplace=True)
    return final

'''
An archive of functions used to scrape the Akron Marathon webpage
'''

from bs4 import BeautifulSoup
import requests as r

race_2021 = 11979
race_2019 = 8097
race_2018 = 6564
race_2017 = 5214

id_year_dict = {
    11979: 2021,
    8097: 2019,
    6564: 2018,
    5214: 2017
}

loc = 0
race = race_2021

def scrape_page(url):
    response = r.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'lxml')
    return soup

def extract_rows_from_page(soup):
    all_rows = soup.find_all(class_='runnersearch-row')
    return all_rows
    
def extract_data_from_row(row):
    row_links = row.find_all('a')
    dataframe = pd.DataFrame({
        'Bib': [row_links[0].text],
        'Name': [row_links[1].text],
        'Sex': [row_links[2].text],
        'Age': [row_links[3].text],
        'City': [row_links[4].text],
        'State': [row_links[5].text],
        'Overall': [row_links[6].text],
        'SexPl': [row_links[7].text],
        'DivPl': [row_links[8].text],
        'Time': [row_links[9].text]
    })
    return dataframe

def scrape_one_page(race, loc):
    dataframe = pd.DataFrame(columns=['Bib','Name','Sex','Age','City','State','Overall','SexPl','DivPl','Time'])
    
    url = f"https://www.mtecresults.com/race/quickResults?raceid={race}&version=223&overall=yes&offset={loc}&perPage=500"
    soup = scrape_page(url)
    all_rows = extract_rows_from_page(soup)

    for row in all_rows:
        row = extract_data_from_row(row)
        dataframe = dataframe.append(row)
    dataframe = dataframe.reset_index(drop=True)
    return dataframe

def get_akron_data():
    df = pd.DataFrame()

    for race in [race_2021, race_2019, race_2018, race_2017]:
        race_df = pd.DataFrame()
        for loc in [0, 500]:
            temp_df = scrape_one_page(race, loc)
            race_df = race_df.append(temp_df)
        race_df['RaceId'] = race
        df = df.append(race_df)

    df = df.reset_index(drop=True)
    df = df.astype({
        'Name':'category',
        'Sex':'category',
        'City':'category',
        'State':'category',
        'Age':float,})

    df['Time'] = pd.to_datetime(df['Time']).apply(lambda x: x.time())
    df['Year'] = df['RaceId'].map(id_year_dict)

    # Some people participated in more than once race,
    # so unique identifiers need to account for that. 
    # Category dtype will categorize each unique value
    df['Name'] = df['Name'].astype('category').cat.codes
    df = df.rename({'Name':'PersonId'}, axis=1)

def save_akron_data():
    import sqlalchemy as s

    engine = s.create_engine('sqlite:///data/marathon.db')
    cnx = engine.connect()

    df.to_sql('akron_marathon', cnx, index=False)
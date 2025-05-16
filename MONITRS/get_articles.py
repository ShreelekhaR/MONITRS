# script to get articles from google search
import pandas as pd
from tqdm import tqdm
import googlesearch



def get_article(search_query):
    
    # retrieve first 5 links from google search
    search_results = googlesearch.search(search_query, num_results=5, unique=True, sleep_interval=5)
    return search_results
            



def main():
    # read the csv file
    df = pd.read_csv('FEMA_filtered.csv', header=0)
    print(f"Number of rows: {len(df)}")
    
    # create a search query for all rows
    queries = []
    for idx, row in df.iterrows():
        query = f"{row['declarationTitle']} {row['incidentType']} {row['designatedArea']} {row['state']} {row['incidentBeginDate']}"
        queries.append((idx, query))
    
    print(f"Number of queries: {len(queries)}")

    f = open('articles.csv', 'a+')  # Open the CSV file in append mode

    for i,search_query in tqdm(enumerate(queries)):
        # print(f"---------------------Query {i+1}/{len(queries)}: {search_query[1]}")
       
        links = get_article(search_query[1])
        
        
        for link in links:
            # print(f"{search_query[0]},{link},\n")
            f.write(f"{search_query[0]},{link},\n")
        
        f.flush()
       
    f.close()
   
if __name__ == "__main__":
    main()  # Call the main function to execute the script
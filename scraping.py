#DATA COLLECTION:
#Webscraping and API
headlines_url={}
from bs4 import BeautifulSoup
import requests
#Uses both news API as well as downloads from bbc
api_key= 'd0426a4417f047e6aae3777af6a688df'
url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
response=requests.get(url)
if response.status_code==200:
    news_data=response.json()
    articles=news_data.get('articles',[])
for article in articles:
    title=article.get('title')
    url=article.get('url')
    headlines_url[title]=url

#Since the html file of the web page is not present already, we will first fetch the web page
response=requests.get('https://www.bbc.com/news')
if response.status_code==200:
            #print(response.status_code) #200 indicates that it wa su
    #print(response.status_code) #200 indicates that it wa su
    soup=BeautifulSoup(response.text,'lxml') #using lxml parser.
    headlines=soup.find_all('h2')
    #note that headlines is a list that contains all the headlines
    #storing all the headlines data 
   
    i=1
    for headline in headlines:
        text=headline.get_text()
        url_tag=headline.find_parent('a')
        if url_tag and url_tag.get('href'):
            url=url_tag.get('href')
            if url.startswith('/'):
                url='https://www.bbc.com/'+url
           
            headlines_url[text]=url
        i+=1
    print(headlines_url)

    
#A dictionary containing the headlines and the respective url of key value pairs is stored.
#topic modelling 
#from 2 sources NEWSAPI and bbc
else:
    print("Error")




















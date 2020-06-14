#import bs4
#from urllib.request import urlopen as uReq
#from urllib.request import Request,urlopen
import sys
from bs4 import BeautifulSoup as soup
import pandas as pd
import re
#import time

'''
#better to generare links on the run to use their number when saving CSVs
def url_link_generator(start_num,stop_num):
    url_links=[]
    for i in range(start_num,stop_num+1):
        url_link='https://www.kinopoisk.ru/film/'+str(i)+'/reviews'
        url_links.append(url_link)
    return url_links

url_list=url_link_generator(607,615)
'''



def check_my_IP(proxy, url_link='https://www.whatismybrowser.com/detect/ip-address-location'):
    from bs4 import BeautifulSoup as soup
    import re
    from urllib.request import Request,urlopen,ProxyHandler
    from urllib.request import build_opener, install_opener

    proxy_support = ProxyHandler({'http' : proxy,
                                  'https' :proxy})
    opener = build_opener(proxy_support)
    install_opener(opener)
    req = Request(url_link, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        page_html = urlopen(req, timeout=15).read()
        page_soup = soup(page_html, 'html.parser')
        
        IPs=[]
        for a in range (3,0,-1):
            for b in range (3,0,-1):
                for c in range (3,0,-1):
                    for d in range (3,0,-1):
                         newIP='[0-9]'*a+'\.'+'[0-9]'*b+'\.'+'[0-9]'*c+'\.'+'[0-9]'*d
                         IPs.append(newIP)
        
        for IP in IPs:
            res=re.search(IP,page_soup.text)
            if res:
                myIP=res.group()
                break
    except OSError:
        myIP='IP not retrieved. OSError - bad proxy :'+proxy
        
    except Exception as e:
        myIP='IP not retrieved. '+str(e)+type(e)+' Bad proxy:'+proxy
    finally:
        return myIP






def get_proxy_list():

    from urllib.request import Request,urlopen
    import pandas as pd
    
    url_link='https://free-proxy-list.net/'
    
    req = Request(url_link, headers={'User-Agent': 'Mozilla/5.0'})
    page_html = urlopen(req, timeout=10).read()
    
    dfs = pd.read_html(page_html)
    proxy_df=dfs[0] #taking the table with proxies
    proxy_list=[]
    for i in range(0,proxy_df.shape[0]):
        if proxy_df.iloc[i,6] == 'yes': #check https support
            proxy_list.append(proxy_df.iloc[i,0]+':'+ str(int(proxy_df.iloc[i,1])))
    return proxy_list






def get_proxy_list_2():
    
    from urllib.request import Request,urlopen
    import pandas as pd
    import re
    
    url_link='https://www.hide-my-ip.com/proxylist.shtml'
    
    req = Request(url_link, headers={'User-Agent': 'Mozilla/5.0'})
    page_html = urlopen(req, timeout=10).read()
    page_html=str(page_html)
    
    
    #generate IP variants
    IPs=[] 
    for a in range (3,0,-1):
        for b in range (3,0,-1):
            for c in range (3,0,-1):
                for d in range (3,0,-1):
                     newIP='[0-9]'*a+'\.'+'[0-9]'*b+'\.'+'[0-9]'*c+'\.'+'[0-9]'*d
                     IPs.append(newIP)
    
    #generate port variants
    ports=[]
    for a in range (5,0,-1):
        port='[0-9]'*a
        ports.append(port)
    
    
    #IPs_ports_prots_list=[]
    IP_list=[]
    ports_list=[]
    prots_list=[]
    res=True
    while res:
        for IP in IPs:
            IP_port_prot=IP+'\",\"p\":\"[0-9]..............................................................'
            res=re.search(IP_port_prot,page_html)
            if res:
                myIP_port_prot=res.group()
    #            IPs_ports_prots_list.append(myIP_port_prot)
                page_html=re.sub(myIP_port_prot,'IP_PORT_PROTOCOL_REMOVED',page_html)
                myIP=re.search(IP,myIP_port_prot)
                myIP=myIP.group()
                myIP_port_prot=re.sub(IP,'IP_REMOVED',myIP_port_prot)
                for port in ports:
                    port='\"p\":\"'+port
                    my_port=re.search(port,myIP_port_prot)
                    if my_port:
                        my_port=my_port.group()
                        myIP_port_prot=re.sub(my_port,'PORT_REMOVED',myIP_port_prot)
                        my_port=re.sub('\"p\":\"','',my_port)
                        my_prot=re.search('HTTPS',myIP_port_prot)
                        if my_prot:
                            my_prot='HTTPS'
                        else: my_prot='HTTP'
                        break
                IP_list.append(myIP)
                ports_list.append(my_port)
                prots_list.append(my_prot)
                break
    
    proxy_df = pd.DataFrame({'IP':IP_list,'Port': ports_list,'Protocol':prots_list})
    
    proxy_list=[]
    for i in range(0,proxy_df.shape[0]):
            if proxy_df.iloc[i,2] == 'HTTPS': #check https support
                proxy_list.append(proxy_df.iloc[i,0]+':'+ str(int(proxy_df.iloc[i,1])))

    return proxy_list


proxy_list_1 = get_proxy_list()
proxy_list_2 = get_proxy_list_2()
proxy_list = proxy_list_1+proxy_list_2



def get_review(url_link,proxy):
#    exc='none'
    from urllib.request import Request,urlopen,ProxyHandler
    from urllib.request import build_opener, install_opener

    proxy_support = ProxyHandler({'http' : proxy,
                                  'https' :proxy})
    opener = build_opener(proxy_support)
    install_opener(opener)
    req = Request(url_link, headers={'User-Agent': 'Mozilla/5.0'})
    page_html='Page not retrieved'
#    success=False
    capcha=False
    exc_mes='OK'
    reviews_df='Empty'
    page_soup='Empty'


    try:
    #    req = Request(url_link, headers={'User-Agent': 'Mozilla/5.0'})
        page_html = urlopen(req, timeout=15).read()
    #        uClient = uReq(url_link)
    #        page_html = uClient.read()
    #        time.sleep(30)
    #        uClient.close()
        page_soup = soup(page_html, 'html.parser')
        movie_title=page_soup.title.text
        if movie_title=='Ой!':
            capcha=True
        #cleaning '— отзывы и рецензии — КиноПоиск' from review
        movie_title=re.sub('— отзывы и рецензии — КиноПоиск',' ',movie_title)
        movie_reviews = page_soup.findAll('div',attrs={'class':'brand_words'})
    
        
        titles=[]
        reviews=[]
        review_rates=[]
        
        for movie_review in movie_reviews:
            movie_review = movie_review.text
            if len(movie_review) > 150: #check if review is not to short
                review_rate = 146 #not rated in the review
                rated=re.search('10 из 10',movie_review)
                if rated:
                    movie_review=re.sub('10 из 10','',movie_review)
                    review_rate=10
                else:
                    rated = re.search('[0-9],[0-9] из 10',movie_review)
                    if rated:
                        rated = re.sub(',','.',rated.group())
                        rated = re.search('[0-9].[0-9]',rated)
                        review_rate = round(float(rated.group())) - 1
                    else:
                        rated=re.search('[0-9] из 10',movie_review)
                        if rated:
                            movie_review=re.sub('[0-9] из 10','',movie_review)
                            review_rate=int(rated.group(0)[0])
                if review_rate != 146:
                    titles.append(movie_title)
                    movie_review=re.sub('\"',' ',movie_review) #cleaning " from review
                    reviews.append(movie_review)
                    review_rates.append(review_rate)
                    reviews_df = pd.DataFrame({'Title':titles,'Review':reviews,'Rate':review_rates})
#                    reviews_df.to_csv('reviews'+str(num)+'.csv', index=False, encoding='utf-8')
                    
    except Exception as exc:
        print('get_review got error:',exc)
        exc_mes=str(exc)
    finally:
        return reviews_df, page_soup,capcha, exc_mes





last_proxy_num=0
start_num=int(sys.argv[1])
end_num=int(sys.argv[2])

for movie_num in range(start_num,end_num+1):
    url_link='https://www.kinopoisk.ru/film/'+str(movie_num)+'/reviews'
    print('\nTrying url: ',url_link)
    for i in range(last_proxy_num,1000):

        proxy=proxy_list[i]
        print('\nTrying proxy: ',proxy)
#        IPaddress=check_my_IP(proxy)
        reviews_df, page_soup, capcha, exc = get_review(url_link,proxy)
        if capcha:
            print('Capcha!')
        
        if i ==(len(proxy_list)-1): #append new proxies to the proxy list
            print('retrieving new proxies')
            try:
                new_proxy_list = get_proxy_list()
                print('get_proxy_list OK')
            except:
                print('get_proxy_list FAILED')
                print('trying_get_proxy_list_2')
                new_proxy_list = get_proxy_list_2()
                
            finally:
                proxy_list = proxy_list + new_proxy_list
      
        
        
        if (capcha==False) and (exc=='OK'):
            if not(type(reviews_df)==str):
                reviews_df.to_csv('reviews'+str(movie_num)+'.csv', index=False, encoding='utf-8')
            else:
                f=open('reviews'+str(movie_num)+'.csv','w')
                f.write('No reviews here')
                f.close()  
            last_proxy_num=i+1
            break
        elif (capcha==False) and (exc=='HTTP Error 404: Not Found'):
            f=open('reviews'+str(movie_num)+'.csv','w')
            f.write('404')
            f.close()
            last_proxy_num=i+1
            break
#    results_df.loc[len(results_df)]=[reviews_df, page_soup, capcha, exc, IPaddress]

print('All done')









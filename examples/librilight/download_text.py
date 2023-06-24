import json
import os
from tqdm import tqdm
from urllib import request
from urllib.request import Request, urlopen
import random
import re
import sys

def validate_filename(f: str):
    f = f.replace('?','')
    f = f.replace('/"', "")
    f = f.replace('/','')
    f = f.replace('\"','')
    return f

def gutenberg_preprocess(link: str) -> str:
    # process gutenberg link
    headers = ["Chrome/104.0.5112.79", "Mozilla/5.0"]
    assert "gutenberg" in link, link
    link = link.rsplit('#', 1)[0]
    if link.endswith(("txt", "html","htm")):
        return link
    elif "files/" in link:
        link = link.rstrip('/') # remove trailing 
        book_id = link.split('/')[-1]
        pattern = re.compile(f"{book_id}.*.txt")
        return link + f'/{pattern[0]}'
        
    else:
        req = Request(url=link, headers={'User-Agent': headers[random.randint(0,1)]}) # This prevents 403 error
        html = urlopen(req, timeout=30).read().decode('utf-8')
        start = html.find("Plain Text UTF-8")
        rough_range = html[start:].split('</td>', 2)[1]
        url = rough_range.rsplit('>', 1)[-1]
        assert "gutenberg" in url, url 

        return url
    
def archive_proprocess(link: str) -> str:
    
    headers = ["Chrome/104.0.5112.79", "Mozilla/5.0"]
    assert "archive.org" in link, link
    link = link.rsplit('#', 1)[0]
    link = link.rstrip('/')
    req = Request(url=link, headers={'User-Agent': headers[random.randint(0,1)]}) # This prevents 403 error
    html = urlopen(req, timeout=30).read().decode('utf-8')
    start = html.find("FULL TEXT")
    
    if start == -1:
        return link
    
    rough_range = html[start-1000:start].split('href="', 2)[-1]
    url = rough_range.split("\"", 1)[0]
    
    pre_fix = "https://archive.org"
    url = pre_fix + url
    
    return url

def ccel_proprocess(link: str) -> str:
    headers = ["Chrome/104.0.5112.79", "Mozilla/5.0"]
    assert "ccel.org" in link, link
    link = link.rsplit('#', 1)[0]
    link = link.rstrip('/')
    req = Request(url=link, headers={'User-Agent': headers[random.randint(0,1)]}) # This prevents 403 error
    html = urlopen(req, timeout=30).read().decode('utf-8')
    start = html.find("Unicode Text (utf-8)")
    
    if start == -1:
        return link
    
    rough_range = html[start-500:start].split('href="')[-1]
    url = rough_range.split("\"", 1)[0]
    
    pre_fix = "https://ccel.org"
    url = pre_fix + url
    
    return url

def download_text(subset):
    print(f"Downloading for {subset}")
    js = f"book_links_{subset}processed.json"
    output_folder = f"output_text_{subset}_with_subfolder"
    headers = ["Chrome/104.0.5112.79", "Mozilla/5.0"]
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print(f"Text will be stored to {output_folder}")
    
    failed_books = {}
    
    with open(js, 'r') as f:
        data = json.load(f)

    for book in tqdm(data):

        urls = data[book]
        book = validate_filename(book) # validate name of book
        book = re.sub(r'[^\w\s]', '', book)
        
        book_folder = f"{output_folder}/{book}"
        if not os.path.exists(book_folder):
            os.mkdir(book_folder)
        
        for i, url in enumerate(urls):
            url = url.rstrip() # remove the trailing control bytes
            url = url.rstrip('/')
            
            out_txt = f"{book_folder}/{book}-{i}.txt" # the target directory
        
            if os.path.exists(out_txt): # already stored
                continue     
            try:
                orig_url = url
                if "archive.org" in url:
                    url = archive_proprocess(orig_url)
                elif "ccel.org" in url and url.endswith(".html"):
                    url = ccel_proprocess(orig_url)
                req = Request(url=url, headers={'User-Agent': headers[random.randint(0,1)]}) # This prevents 403 error
                response = urlopen(req, timeout=20).read()
                with open(out_txt, 'wb') as fout:
                    fout.write(response)
            except KeyboardInterrupt:
                # quit
                sys.exit()
            except:
                if book in failed_books:
                    failed_books[book].append(url)
                else:
                    failed_books[book] = [url]
                print(f"URL is not valid: {url}. Original url: {orig_url}")
        print(f"Finished for book: {book}")
        
    with open(f"failed_{subset}_after_recover_archive.json", 'w') as fout:
        json.dump(failed_books, fout, indent=4)
    
    
def _download_text_deprecated(subset):
    #subset = "large"
    print(f"Downloading for {subset}")
    js = f"book_links_{subset}processed.json"
    output_folder = f"output_text_{subset}_with_subfolder"
    headers = ["Chrome/104.0.5112.79", "Mozilla/5.0"]
    
    failed_books = {}

    with open(js, 'r') as f:
        data = json.load(f)

    for book in tqdm(data):

        urls = data[book]
        response_list = []
        book = validate_filename(book) # validate name of book
        book = re.sub(r'[^\w\s]', '', book)
        
        book_folder = f"{output_folder}/{book}"
        if not os.path.exists(book_folder):
            os.mkdir(book_folder)
        
        for i, url in enumerate(urls):
            url = url.rstrip() # remove the trailing control bytes
            url = url.rstrip('/')
            
            out_txt = f"{book_folder}/{book}-{i}.txt" # the target directory
        
            if os.path.exists(out_txt): # already stored
                continue
            
            try:
                req = Request(url=url, headers={'User-Agent': headers[random.randint(0,1)]}) # This prevents 403 error
                response = urlopen(req, timeout=60).read()
                response_list.append(response)
                
                with open(out_txt, 'wb') as fout:
                    for text in response_list:
                        fout.write(text)
            except KeyboardInterrupt:
                # quit
                sys.exit()
            except:
                if book in failed_books:
                    failed_books[book].append(url)
                else:
                    failed_books[book] = [url]
                print(f"URL is not valid: {url}. Original url: {url}")
        
            
        print(f"Finished for book: {book}")
        
    with open(f"failed_{subset}.json", 'w') as fout:
        json.dump(failed_books, fout, indent=4)
        

if __name__=="__main__":
    for subset in ["small", "medium", "large"]:
        download_text(subset=subset)

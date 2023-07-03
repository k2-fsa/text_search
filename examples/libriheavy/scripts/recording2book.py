import json
import glob
from tqdm import tqdm
import re
import os

def get_mapping(subset: str):
    libri_light_root = "libri-light/"
    text_root = f"output_text_{subset}_cleaned"
    flac_files = glob.glob(libri_light_root + subset + "/*/*/*.flac")
    
    output_js = f"recording2book_{subset}.json"
    js = {}
    
    failed_books = set()
    
    #import pdb; pdb.set_trace()
    for flac in tqdm(flac_files):
        key = flac.replace(libri_light_root, "")[:-5]
        meta_js = flac.replace(".flac", ".json")
        with open(meta_js, 'r') as f:
            meta_data = json.load(f)["book_meta"]
            
        book_name = meta_data["title"]
        book_name = re.sub(r'[^\w\s]', '', book_name)
        
        if not os.path.isdir(text_root + "/" + book_name):
            print(f"Book: {book_name} is not found!")
        #assert os.path.isdir(text_root + "/" + book_name), book_name
        target_txt = text_root + "/" + book_name + "/text.txt"
        
        if os.path.isfile(target_txt):
            js[key] = target_txt
        else:
            js[key] = ""
            if book_name not in failed_books:
                failed_books.add(book_name)
                print(f"No book found for {book_name}")
    
    print(f"A total of {len(failed_books)} empty folders!")
    with open(output_js, 'w') as f:
        json.dump(js, f, indent=4, ensure_ascii=False)
        
if __name__=="__main__":
    subset = "large"
    print(f"Processing dataset: {subset}")
    get_mapping(subset=subset)
        
        
        
        
        
    
    
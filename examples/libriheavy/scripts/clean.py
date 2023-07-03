import glob
from tqdm import tqdm
import os
from bs4 import BeautifulSoup

from process import replace_control_char

def clean_file(txt: str):
    soup = BeautifulSoup(txt, 'html.parser')
    text = soup.get_text(separator=" ", strip=True)
    return text

def post_processing(text: str):
    splitted = text.split() # split for all types of whitespace
    N = len(splitted)
    for i in range(N):
        s = replace_control_char(splitted[i])
        splitted[i] = s
    return " ".join(splitted)
    
def clean_dataset(subset: str):
    txt_files = glob.glob(f"output_text_{subset}/output_text_{subset}_with_subfolder/*/*.txt")
    for txt in tqdm(txt_files):
        #import pdb; pdb.set_trace()
        with open(txt, 'rb') as f:
            data = f.read()
        clean_text = clean_file(data) # clean html file
        clean_text = post_processing(clean_text)
        
        _, book_name, id = txt.rsplit("/", 2)
        dst_folder = f"output_text_{subset}_cleaned/{book_name}/"
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)
        dst_file = f"output_text_{subset}_cleaned/{book_name}/{id}"
        with open(dst_file, 'w') as fout:
            fout.writelines(clean_text)
            
def clean_dataset_v2(subset: str):
    import os
    root = f"output_text_{subset}/output_text_{subset}_with_subfolder"
    book_folder = os.listdir(root)
    for folder in tqdm(book_folder):
        dst_folder = f"output_text_{subset}_cleaned/{folder}/"
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)
        
        cur_root = root + "/" + folder
        txts = glob.glob(cur_root + '/*.txt')
        if len(txts) == 0:
            continue
        txts = sorted(txts, key=lambda x: os.stat(x).st_size, reverse=True)
        txt = txts[0] # the largest file should be the one containing all the text
        
        with open(txt, 'rb') as f:
            data = f.read()
        clean_text = clean_file(data) # clean html file
        clean_text = post_processing(clean_text)
        
        _, book_name, _ = txt.rsplit("/", 2)
        dst_folder = f"output_text_{subset}_cleaned/{book_name}/"
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)
        dst_file = f"output_text_{subset}_cleaned/{book_name}/text.txt"
        with open(dst_file, 'w') as fout:
            fout.writelines(clean_text)
    
    
if __name__=="__main__":
    subset = "large"
    print(f"Processing dataset: {subset}")
    clean_dataset_v2(subset)
    
    
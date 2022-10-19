import pandas as pd
import glob
import requests
import os
from datetime import datetime


def get_img(url):
    # get response
    response = requests.get(url)

    name_img = os.path.basename(url)
    datestr = name_img.split('_')[-1].split('.')[0]
    dateobj = datetime.strptime(datestr, "%Y%m%d%H")

    path = f"./{dateobj.strftime('%Y/%j')}/"
    os.makedirs(path, exist_ok=True)

    # save in:
    img_out = f"{path}{name_img}"
    print(img_out)

    # save file
    with open(img_out, 'wb') as f:
        f.write(response.content)
    return


if __name__ == '__main__':
    paths = glob.glob("/home/marten/Desktop/workdir/selenium_work/data/*/250hPa/*.csv")
    url_charts = list(sorted(pd.concat([pd.read_csv(path) for path in paths]).iloc[:, 9].to_list()))
    for url in url_charts:
        get_img(url)
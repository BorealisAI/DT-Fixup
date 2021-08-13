# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


#  The link is provided by the author of IRNet: https://github.com/microsoft/IRNet/issues/6
download_file_from_google_drive('1LgyjtDmf3Xd1txwq8HwKD6d6VJj5pLmn', 'conceptNet.zip')
os.system('unzip conceptNet.zip')
os.system('rm conceptNet.zip')
os.system('mv conceptNet concept_net')

#  The link is provided by the author of Spider
download_file_from_google_drive('1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0', 'spider.zip')
os.system('unzip spider.zip')
os.system('rm spider.zip')

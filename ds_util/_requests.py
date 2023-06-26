import requests
import json
import pandas as pd
from json import JSONDecodeError


def request_to_data(request, return_json=False):
    result_str = request.content.decode()

    if not return_json:
        try:
            result = pd.read_json(request.json()['data'])  # From LTJDS API
        except (TypeError, JSONDecodeError) as e:
            result = pd.DataFrame(request.json())

    else:
        result = json.loads(result_str)

    return result


def get_api_data(url, route, return_json=False):
    request = requests.get(url + route)
    return request_to_data(request, return_json)


def post_api_data(url, route, json_payload, headers=None, return_json=False, flask_d3=False):
    if headers is None:
        headers = {"Content-Type": "application/json;charset=UTF-8"}

    if flask_d3 is not None:
        json_payload = {'json_payload': json.dumps(json_payload)}

    request = requests.post(url + route,
                            headers=headers,
                            json=json_payload)

    return request_to_data(request, return_json)


def list_html_tables(url, read_html_kws=None):
    r = requests.get(url)

    if read_html_kws is None:
        read_html_kws = {}

    df_list = pd.read_html(r.text, **read_html_kws)

    return df_list

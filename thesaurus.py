import json
import requests

def _handle_response(data, load, return_response_obj):
    if data.text.strip() is "":
        if data.status_code == 404:
            raise ErrorCode404("Not Found: No data could be found for the word or alternates")
            return
        elif data.status_code == 500:
            raise ErrorCode500("Usage Exceeded: Usage limits have been exceeded, Inactive key: The key is not active, Missing words: No word was submitted")
            return
    if return_response_obj is False:
        if load is False:
            return data.text
        elif load is True:
            return json.loads(data.text)
    elif return_response_obj is True:
        if load is False:
            obj = Response(data.text, data.status_code)
            return obj
        elif load is True:
            obj = Response(json.loads(data.text), data.status_code)
            return obj

class ErrorCode404(Exception):
    pass

class ErrorCode500(Exception):
    pass

class Response:
    def __init__(self, data, status_code):
        self.data = data
        self.status_code = status_code

class Thesaurus:
    def __init__(self, key: str, version: int = 2):
        self.key = key
        self.vers = version

    def rawRequest(self, word, format="json"):
        r = requests.get("http://words.bighugelabs.com/api/{0.vers}/{0.key}/{1}/{2}".format(self, word, format))
        return r

    def rawText(self, word, format="json"):
        r = requests.get("http://words.bighugelabs.com/api/{0.vers}/{0.key}/{1}/{2}".format(self, word, format))
        return _handle_response(r, False, False)

    def loadedObj(self, word):
        r = requests.get("http://words.bighugelabs.com/api/{0.vers}/{0.key}/{1}/json".format(self, word))
        return _handle_response(r, True, True)

    def rawObj(self, word):
        r = requests.get("http://words.bighugelabs.com/api/{0.vers}/{0.key}/{1}/json".format(self, word))
        return _handle_response(r, False, True)
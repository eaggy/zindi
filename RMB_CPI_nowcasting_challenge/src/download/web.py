# -*- coding: utf-8 -*-
"""
The file contains python code of basic web downloader.

Created on 26.09.2023

@author: ihar
"""

import platform
import requests
from requests.auth import HTTPProxyAuth
from requests.packages import urllib3
from settings import USE_PROXY, PROXY_USER, PROXY_PASSWORD
from settings import PROXY_URL, PROXY_PORT


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


HEADERS = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.5",
    "sec-fetch-site": "same-origin",
}


class WebLoader:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers = {"user-agent": self.__build_user_agent_string()}
        self.session.headers.update(HEADERS)
        self.session.proxies = {}
        self.session.auth = None
        self.session.trust_env = False
        self.session.verify = False
        self.__set_proxy()

    @staticmethod
    def __build_user_agent_string() -> str:
        """Build the OS specific user agent string.

        Returns:
            User agent string.
        """
        user_agent_string = ""
        system = platform.system()
        if system == "Windows":
            user_agent_string = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                                "AppleWebKit/537.36 (KHTML, like Gecko) " \
                                "Chrome/110.0.0.0 Safari/537.36"
        if system == "Linux":
            user_agent_string = "Mozilla/5.0 (X11; Linux x86_64) " \
                                "AppleWebKit/537.36 (KHTML, like Gecko) " \
                                "Chrome/110.0.0.0 Safari/537.36"
        return user_agent_string

    def __set_proxy(self):
        """Set proxy settings if required.

        """
        if USE_PROXY:
            self.session.proxies = {
                "http": f"http://{PROXY_USER}:{PROXY_PASSWORD}@{PROXY_URL}:{PROXY_PORT}",
                "https": f"http://{PROXY_USER}:{PROXY_PASSWORD}@{PROXY_URL}:{PROXY_PORT}"
            }
            self.session.auth = HTTPProxyAuth(PROXY_USER, PROXY_PASSWORD)

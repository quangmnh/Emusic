#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================================
Python script for Spotify API
==================================================================
    AUTHOR: quangmnh
    CREATED: 2021, Sep 5th
    UPDATED: 2021, Sep 28th
    PURPOSE: Functions for getting meta data from Spotify API
    LICENSE: Copyleft
------------------------------------------------------------------
    VERSION: 1.0.2
    Revison History:
        2021, Sep 5th: Initialized, basic
            functions (get track,multi tracks, audio features, 
            analysis)
        2021, Sep 15th: Added the search function
        2021, Sep 28th: Added user authorization flow
------------------------------------------------------------------
    TO DO:
            Build a server for App hosting cause the API need to 
        be redirected somehow
            Deal with the unicode character in the query result
==================================================================
"""


import base64
import re
from typing import List
from requests.models import requote_uri
import six
import requests
# import logging
import ast
import json
import urllib.parse
import winreg
import webbrowser
import sys 
from urllib.parse import urlparse
from urllib.parse import parse_qs
import time
import hashlib
from ftfy import fix_encoding,guess_bytes,fix_text
class Spotify:
    def __init__(self, _VERBOSE = False):
        self._session = requests.session()
        self.play_list = []
        self.authorize_url = "https://accounts.spotify.com/api/token"
        self.authorize_user = "https://accounts.spotify.com/authorize"
        self.api_url = "https://api.spotify.com/v1/"
        self.verbose = _VERBOSE
        self.market = "VN"
        self.default_type = ["album" , "artist", "playlist", "track", "show", "episode"]
        self.install_path = "D:/BT/211/Thesis/GitRepos/Emusic/API/Spotify/"
        self.redirect_uri = "emusic%3A%2F%2Fabc.com"
        self.user_authorized = False
        self.state = "emusicstate"
        self.isRefreshable = False
        self.redirect_command = "python " + self.install_path + "tesst.py"+" %1"
    

    def url_protocol_handler_register(self):
        """
        Create a protocol handler for user, triggering the open of app from self.install_path

        to be updated while dev-ing the app
        """
        key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, "emusic")
        # winreg.SetValueEx(key,__value='@="Spotify API call back""URL Protocol"=""')
        winreg.SetValueEx(key,'', 1,winreg.REG_SZ,"Emusic App")
        winreg.SetValueEx(key,'URL Protocol', 1, winreg.REG_SZ, "")
        shell = winreg.CreateKey(key, "shell")
        open = winreg.CreateKey(shell, "open")
        command = winreg.CreateKey(open, "command")
        winreg.SetValueEx(command, '', 0, winreg.REG_SZ, self.redirect_command)
    
    def code_generator(self, key:str):
        code_verifier = base64.b64encode(six.text_type(hashlib.sha256(key)).encode("ascii"))
        return code_verifier.decode("ascii")

    def user_authorize_pkce(self, client_id, key):
        """
        Initialize user credential authorization flow with PKCE

        :param client_id: App's client_id, provided in developer mode 

        :return: No but open the app again on uri redirection
        """
        self.client_id = client_id
        if self.verbose:
            print("User authorizing with PKCE: ")
        # header = self._make_authorization_headers(client_id,client_secret)
        # body = {'grant_type' : 'client_credentials'}
        url = self.authorize_user + "?client_id=" + str(client_id) + "&response_type=code&redirect_uri=" + str(self.redirect_uri) +"&code_challenge_method=S256&code_challenge=" +self.code_generator(self.get_key())+ "&state=emusicstate&scope=user-read-private%20user-read-email"
        # response = self._session.post(url=url)
        webbrowser.open_new_tab(url)
        if self.verbose:
            print("     Request sent, waiting for redirection")

    def get_key(self):
        self.key = "alksdlasndlsandlnsaldknsf.___,,nalsfba116sdbfkasbdvjkgasmbvglasdjvgbaksgv"
        return self.key
    def check_user_credential_pkce(self,client_id):
        """
        Check user credential authorization, code received from the uri.
        :return: Bool value indicating if app open via redirection.
        """
        if len(sys.argv[1:])>0:
            temp = sys.argv[1:]
        # if len(url) > 0:
        #     temp = url
            parsed_url = urlparse(temp[0])
            state = parse_qs(parsed_url.query)['state'][0]
            if state!=self.state:
                if self.verbose:
                    print("     State invalid, warning cross-site attack")
                return False
            try:
                self.code = parse_qs(parsed_url.query)['code'][0]
                self.user_authorized = True
                if self.verbose:
                    print("     User Authorized w/ pkce, initiating access token request")
                self.request_user_access_token_pkce(self,client_id)
                return True    
            except:
                err = parse_qs(parsed_url.query)['error'][0]
                if self.verbose:
                    print("     Error authorizing: {}".format(err))
                return False
        else:
            self.user_authorized = False
            return False    

    def request_user_access_token_pkce(self, client_id):
        """
        Request access token using user authorization code flow, with pkce
        No client secret provide so, maybe, safer???

        :param client_id: App's client_id, provided in developer mode 
        :param client_secret: App's client secret, provided in developer mode
        :return int: 0 mean successfully request the access_token
        """
        self.client_id = client_id

        if self.verbose:
            print("User code authorizing with pkce: ")
        body = {'client_id' : str(client_id), 'grant_type' : 'authorization_code', 'code': str(self.code), 'redirect_uri': str(self.redirect_uri), 'code_verifier' : str(self.code_generator(self.get_key()))}
        response = self._session.post(url=self.authorize_url, data=body)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
            print("     Content: " + str(response.content))
        if response.status_code == 200:
            self.user_authorized_code = ast.literal_eval(response.content.decode("ascii"))   
            self.access_token = self.user_authorized_code["access_token"]
            self.scope = self.user_authorized_code["scope"]
            self.expires_in = self.user_authorized_code["expires_in"]
            self.token_start = time.time()
            self.refresh_token = self.user_authorized_code["refresh_token"]
            self.isRefreshable = True
            if self.verbose:
                print("     Access Token: " + self.access_token)
            return 0
        else:
            print("     Error in user authorization:")
            print(response)
            return 1

    def check_user_credential(self, client_id, client_secret):
        """
        Check user credential authorization, code received from the uri.
        :return: Bool value indicating if app open via redirection.
        """
        if len(sys.argv[1:])>0:
            temp = sys.argv[1:]
        # if len(url) > 0:
        #     temp = url
            parsed_url = urlparse(temp[0])
            state = parse_qs(parsed_url.query)['state'][0]
            if state!=self.state:
                if self.verbose:
                    print("     State invalid, warning cross-site attack")
                return False
            try:
                self.code = parse_qs(parsed_url.query)['code'][0]
                self.user_authorized = True
                if self.verbose:
                    print("     User Authorized, initiating access token request")
                self.request_user_access_token(client_id,client_secret)
                return True    
            except:
                err = parse_qs(parsed_url.query)['error'][0]
                if self.verbose:
                    print("     Error authorizing: {}".format(err))
                return False
        else:
            self.user_authorized = False
            return False

    def user_authorize(self, client_id):
        """
        Initialize user credential authorization flow

        :param client_id: App's client_id, provided in developer mode 

        :return: No but open the app again on uri redirection
        """
        self.client_id = client_id
        if self.verbose:
            print("User authorizing: ")
        # header = self._make_authorization_headers(client_id,client_secret)
        # body = {'grant_type' : 'client_credentials'}
        url = self.authorize_user + "?client_id=" + str(client_id) + "&response_type=code&redirect_uri=" + str(self.redirect_uri) + "&scope=user-read-private%20user-read-email&state=emusicstate"
        # response = self._session.post(url=url)
        webbrowser.open_new_tab(url)
        if self.verbose:
            print("     Request sent, waiting for redirection")
    def request_user_access_token(self, client_id, client_secret):
        """
        Request access token using user authorization code flow

        :param client_id: App's client_id, provided in developer mode 
        :param client_secret: App's client secret, provided in developer mode
        :return int: 0 mean successfully request the access_token
        """
        self.client_id = client_id
        self.client_secret = client_secret

        if self.verbose:
            print("User code authorizing: ")
        header = self._make_authorization_headers(client_id,client_secret)
        body = {'grant_type' : 'authorization_code', 'code': str(self.code), 'redirect_uri': str(self.redirect_uri)}
        response = self._session.post(url=self.authorize_url, data=body,headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
            print("     Content: " + str(response.content))
        if response.status_code == 200:
            self.user_authorized_code = ast.literal_eval(response.content.decode("ascii"))   
            self.access_token = self.user_authorized_code["access_token"]
            self.scope = self.user_authorized_code["scope"]
            self.expires_in = self.user_authorized_code["expires_in"]
            self.token_start = time.time()
            self.refresh_token = self.user_authorized_code["refresh_token"]
            self.isRefreshable = True
            if self.verbose:
                print("     Access Token: " + self.access_token)
            return 0
        else:
            print("     Error in user authorization:")
            print(response)
            return 1
    def refresh_user_access_token(self, client_id, client_secret):
        """
        Request access token using user authorization code flow

        :param client_id: App's client_id, provided in developer mode 
        :param client_secret: App's client secret, provided in developer mode
        :return int: 0 mean successfully request the access_token
        """
        self.client_id = client_id
        self.client_secret = client_secret

        if not self.isRefreshable:
            if self.verbose:
                print("     Authorization can't be refreshed, did you authorize it 1st?")
            return 1
        if self.verbose:
            print("Refreshing access token: ")
        header = self._make_authorization_headers(client_id,client_secret)
        body = {'grant_type' : 'refresh_token', 'refresh_token': str(self.refresh_token), }
        response = self._session.post(url=self.authorize_url, data=body,headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
            print("     Content: " + str(response.content))
        if response.status_code == 200:
            self.user_authorized_code = ast.literal_eval(response.content.decode("ascii"))   
            self.access_token = self.user_authorized_code["access_token"]
            self.scope = self.user_authorized_code["scope"]
            self.expires_in = self.user_authorized_code["expires_in"]
            self.token_start = time.time()
            self.refresh_token = self.user_authorized_code["refresh_token"]
            self.isRefreshable = True
            if self.verbose:
                print("     Access Token: " + self.access_token)
            return 0
        else:
            print("     Error in user authorization:")
            print(response)
            return 1
    def _make_authorization_headers(self, client_id, client_secret):
        """
        Make authorization header for the API, which requires client id and client secret to 
        be authorized in app credential flow, which then are required to be a base64-encoded
        ASCII string

        :param client_id: App's client_id, provided in developer mode 
        :param client_secret: App's client secret, provided in developer mode
        :return Tuple: a tuple contain client credential authorizing heaader  
        """
        
        auth_header = base64.b64encode(six.text_type(client_id + ":" + client_secret).encode("ascii"))
        return {"Authorization": "Basic %s" % auth_header.decode("ascii")}

    def app_authorize(self, client_id, client_secret):
        """
        Authorie the app with client credential flow, which send a HTTP post request
        to the API, contain the client id and client secret to get the access token 
        required for accessing the metadata

        :param client_id: App's client_id, provided in developer mode 
        :param client_secret: App's client secret, provided in developer mode
        :return Tuple: a tuple contain the access token, saved in access_token  
        """

        self.client_id = client_id
        self.client_secret = client_secret

        if self.verbose:
            print("App authorizing: ")
        header = self._make_authorization_headers(client_id,client_secret)
        body = {'grant_type' : 'client_credentials'}
        response = self._session.post(url=self.authorize_url, data=body,headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
            print("     Content: " + str(response.content))
        if response.status_code == 200:
            self.client_credential = ast.literal_eval(response.content.decode("ascii"))   
            self.access_token = self.client_credential["access_token"]
            if self.verbose:
                print("     Access Token: " + self.access_token)
            return 0
        else:
            print("     Error in app authorization:")
            print(response)
            return 1

    def get_track(self, track_id):
        """
        Send a HTTP get request to request track information of a track, indentified by track ID

        :param track_id: Track's ID, can be get from the search function below 
        :return Tuple: a tuple contain track information, example might be found in 
        returnsample/get_track.txt 
        """
        if self.verbose:
            print("Track query in process ... ")
        header = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer '+str(self.access_token)}
        url = self.api_url + "tracks/" + str(track_id)+"?market="+str(self.market)
        response = self._session.get(url = url, headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
        if response.status_code == 200:
            track = response.content.decode("ascii")   
            if self.verbose:
                print("     Track information: " + track)
            return json.loads(track)
        elif response.status_code == 401:
            if self.verbose:
                print("     Session expired, Re-authorizing .....")
            self.app_authorize(self.client_id, self.client_secret)
            if self.verbose:
                print("Re-query track")
            self.get_track(track_id)
        else:
            print("     Error in track query:")
            print(response)
            return 1

    def get_tracks_from_playlist(self, playlist_id):
        """
        Send a HTTP get request to request track ids from a playlist

        :param playlist_id: Playlist's ID, can be get from the search function below 
        :return Tuple: a tuple contain track information, example might be found in 
        returnsample/get_track.txt 
        """
        if self.verbose:
            print("Track query in process ... ")
        header = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer '+str(self.access_token)}
        url = self.api_url + "playlists/" + str(playlist_id)+"?market="+str(self.market)+"&fields=tracks.items(track(id%2Cname))"
        response = self._session.get(url = url, headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
        if response.status_code == 200:
            try:
                tracks = response.content.decode("utf-8")   
            except:
                tracks = response.content.decode("ascii")
            if self.verbose:
                print("     Track information: " + tracks)
            return json.loads(tracks)
        elif response.status_code == 401:
            if self.verbose:
                print("     Session expired, Re-authorizing .....")
            self.app_authorize(self.client_id, self.client_secret)
            if self.verbose:
                print("Re-query track")
            self.get_tracks_from_playlist(playlist_id)
        else:
            print("     Error in track query:")
            print(response)
            return 1
    def get_tracks(self, track_ids):
        """
        Send a HTTP get request to request track information of multiple tracks, indentified by track IDs
        Much like get_track but send a request wwith a list of IDs seperated by commas.

        :param track_ids: list type, contain track IDs to be queried. 
        :return Tuple: a tuple with only one key "tracks" in which value contains a list of tuples of track information
        Example might be found in returnsample/get_tracks.txt 
        """
        if self.verbose:
            print("Tracks query in process ... ")
        header = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer '+str(self.access_token)}
        url = self.api_url + "tracks?ids=" + str(track_ids[0])
        for i in range (1,len(track_ids)):
            url += "%2C" + str(track_ids[i])
        url += "&market=" +str(self.market)
        response = self._session.get(url = url, headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
        if response.status_code == 200:
            track = response.content.decode("ascii")   
            if self.verbose:
                print("     Tracks information: " + track)
            return json.loads(track)
        elif response.status_code == 401:
            if self.verbose:
                print("     Session expired, Re-authorizing .....")
            self.app_authorize(self.client_id, self.client_secret)
            if self.verbose:
                print("Re-query tracks")
            self.get_track(track_ids)
        else:
            print("     Error in track query:")
            print(response)
            return 1

    

    def get_track_audio_features(self, track_id):
        """
        Send a HTTP get request to request track audio features of a track, which contains useful information
        for song analysis such as danceability, energy, ... indentified by track ID 

        :param track_id: Track's ID, can be get from the search function below 
        :return Tuple: a tuple contain track's audio feature, example might be found in 
        returnsample/get_track_audio_feature.txt 
        """
        if self.verbose:
            print("Getting audio feature of the track ... ")
        header = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer '+str(self.access_token)}
        url = self.api_url + "audio-features/" + str(track_id)
        response = self._session.get(url = url, headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
        if response.status_code == 200:
            track = response.content.decode("ascii")   
            if self.verbose:
                print("     Track audio features: " + track)
            return json.loads(track)
        elif response.status_code == 401:
            if self.verbose:
                print("     Session expired, Re-authorizing .....")
            self.app_authorize(self.client_id, self.client_secret)
            if self.verbose:
                print("Re-query track")
            self.get_track(track_id)
        else:
            print("     Error in track query:")
            print(response)
            return 1

    def get_tracks_audio_features(self, track_ids):
        """
        Send a HTTP get request to request track audio features of multiple tracks, which contain useful information
        for song analysis such as danceability, energy, ... indentified by track IDs

        :param track_ids: Track's IDs, can be get from the search function below 
        :return Tuple: a tuple contain one key "audio_analysis" in which value contain list of tuples of track's audio features,
        example might be found in returnsample/get_track_audio_feature.txt 
        """
        if self.verbose:
            print("Tracks audio features in process ... ")
        header = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer '+str(self.access_token)}
        url = self.api_url + "audio-features?ids=" + str(track_ids[0])
        for i in range (1,len(track_ids)):
            url += "%2C" + str(track_ids[i])
        # print(url)
        response = self._session.get(url = url, headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
        if response.status_code == 200:
            track = response.content.decode("ascii")   
            if self.verbose:
                print("     Audio features information: " + track)
            return json.loads(track)
        elif response.status_code == 401:
            if self.verbose:
                print("     Session expired, Re-authorizing .....")
            self.app_authorize(self.client_id, self.client_secret)
            if self.verbose:
                print("Re-query tracks")
            self.get_track(track_ids)
        else:
            print("     Error in track query:")
            print(response)
            return 1
    def get_track_audio_analysis(self, track_id):
        """
        Send a HTTP get request to request track audio analysis of a track, which contain doubtedly useful information
        for song analysis such as timbre, beats, pitch ... indentified by track ID. really hope to find a use for them

        :param track_id: Track's ID, can be get from the search function below 
        :return Tuple: a tuple contain track's audio analysis, example might be found in 
        returnsample/get_track_audio_analysis.txt 
        """
        if self.verbose:
            print("Track analysis in process ... ")
        header = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer '+str(self.access_token)}
        url = self.api_url + "audio-analysis/" + str(track_id)
        response = self._session.get(url = url, headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
        if response.status_code == 200:
            track = response.content.decode("ascii")   
            if self.verbose:
                print("     Audio analysis: " + track)
            return json.loads(track)
        elif response.status_code == 401:
            if self.verbose:
                print("     Session expired, Re-authorizing .....")
            self.app_authorize(self.client_id, self.client_secret)
            if self.verbose:
                print("Re-query track")
            self.get_track(track_id)
        else:
            print("     Error in track query:")
            print(response)
            return 1
    def search(self, q:str, type:list, limit: int, offset: int):
        """
        Send a HTTP get request to queries information of some track/album/artist/episode/show
        by using some keywords (no clue about Unicode ones for now) 

        :param q: string of keywords used to query for data
        :param type: list of type of the prefered result, track/album/artist/..., all by default if list is empty
        :param limit: the limit of number of results, min is 1 and max is 50.
        :param offset: offset of results, can be used to queries for the 51st and beyond result, haven't checked yet
        :return Tuple: a tuple contain track's audio analysis, example might be found in 
        returnsample/get_track_audio_analysis.txt 
        """
        if self.verbose:
            print("Searching in process ... ")
        header = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer '+str(self.access_token)}
        url = self.api_url + "search?q=" + urllib.parse.quote(str(q)) + "&type="
        # print(urllib.parse.quote(str(q)))
        if len(type) == 0:
            type = self.default_type
        url+=str(type[0])
        for i in range (1,len(type)):
            url += "%2C" + str(type[i])
        url+="&market=" + str(self.market) + "&limit=" + str(limit) + "&offset=" + str(offset) 
        # print(url)
        response = self._session.get(url = url, headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
        if response.status_code == 200:
            # print(guess_bytes(response.content)[1])
            try:
                res = response.content.decode("utf-8")   
            except:
                try:
                    res = response.content.decode("ascii")
                except:
                    try: 
                        res = response.content.decode("utf-32")
                    except:
                        res = response.content.decode("utf-16")
            if self.verbose:
                print("     Search results: " + res)
            return json.loads(res)
        elif response.status_code == 401:
            if self.verbose:
                print("     Session expired, Re-authorizing .....")
            self.app_authorize(self.client_id, self.client_secret)
            if self.verbose:
                print("Search start again")
            self.search(q, type, limit, offset)
        else:
            print("     Error in searching:")
            print(response)
            return 1
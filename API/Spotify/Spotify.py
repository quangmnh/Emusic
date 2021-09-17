#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================================
Python script for Spotify API
==================================================================
    AUTHOR: quangmnh
    CREATED: 2021, Sep 5th
    UPDATED: 2021, Sep 15th
    PURPOSE: Functions for getting meta data from Spotify API
    LICENSE: Copyleft
------------------------------------------------------------------
    VERSION: 1.0.1
    Revison History:
        2021, Sep 5th: Initialized, basic
            functions (get track,multi tracks, audio features, 
            analysis)
        2021, Sep 15th: Added the search function
------------------------------------------------------------------
    TO DO:
            Build a server for App hosting cause the API need to 
        be redirected somehow
            Implement user authorizing flow
            Deal with the unicode character in the query result
==================================================================
"""


import base64
from typing import List
from requests.models import requote_uri
import six
import requests
# import logging
import ast
import json
import urllib.parse
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
        print(url)
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
        print(urllib.parse.quote(str(q)))
        if len(type) == 0:
            type = self.default_type
        url+=str(type[0])
        for i in range (1,len(type)):
            url += "%2C" + str(type[i])
        url+="&market=" + str(self.market) + "&limit=" + str(limit) + "&offset=" + str(offset) 
        print(url)
        response = self._session.get(url = url, headers=header)
        if self.verbose:
            print("     Status code: " + str(response.status_code))
        if response.status_code == 200:
            res = response.content.decode("ascii",errors="ignore")   
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
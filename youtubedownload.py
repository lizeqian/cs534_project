#!/usr/bin/python

from apiclient.discovery import build
from apiclient.errors import HttpError
import argparse
from pytube import YouTube
import os

# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
DEVELOPER_KEY = "AIzaSyBPsK8otoxQqTj_XyAYyKZmL49QM-MmO9U"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def youtube_search(options):
      youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
        developerKey=DEVELOPER_KEY)
    
      # Call the search.list method to retrieve results matching the specified
      # query term.
      search_response = youtube.search().list(
        q=options.q,
        part="id,snippet",
        maxResults=options.max_results,
        pageToken=options.nextpage
      ).execute()
    
      videos = []
      channels = []
      playlists = []
      nextpage = search_response.get("nextPageToken")
    
      # Add each result to the appropriate list, and then display the lists of
      # matching videos, channels, and playlists.
      for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
          videos.append(search_result["id"]["videoId"])
        elif search_result["id"]["kind"] == "youtube#channel":
          channels.append("%s (%s)" % (search_result["snippet"]["title"],
                                       search_result["id"]["channelId"]))
        elif search_result["id"]["kind"] == "youtube#playlist":
          playlists.append("%s (%s)" % (search_result["snippet"]["title"],
                                        search_result["id"]["playlistId"]))
        
      return videos, nextpage
  
def youtube_download(name, number, resolution, dest): #(string, num_of_videos, string, string)
      
      num_dl_videos = 0
      parser = argparse.ArgumentParser(prog='PROG', conflict_handler='resolve')
      parser.add_argument("--q", help="Search term", default=name)
      parser.add_argument("--max-results", help="Max results", default=50)
      parser.add_argument("--nextpage", help="nextpage", default=None)
      args = parser.parse_args()
    
      try:
        search_result, nextpage=youtube_search(args)
      except HttpError, e:
        print "An HTTP error %d occurred:\n%s" % (e.resp.status, e.content)
        
      link_ids=[]
      link_ids=search_result
      
      if not os.path.exists(dest):
          os.makedirs(dest)
      
      for element in link_ids:
          source = "http://www.youtube.com/watch?v="+element
          yt = YouTube(source)
          stream = yt.streams.filter(subtype='mp4', res="480p").first()
          stream.download(output_path=dest)
          num_dl_videos += 1
          print "%d Video Dowloaded\n" %(num_dl_videos)

      while num_dl_videos < number:
          parser.add_argument("--nextpage", help="nextpage", default=str(nextpage))
          args = parser.parse_args()
          try:
            search_result, nextpage=youtube_search(args)
          except HttpError, e:
            print "An HTTP error %d occurred:\n%s" % (e.resp.status, e.content)
          
          link_ids=[]
          link_ids=search_result
      
          for element in link_ids:
              source = "http://www.youtube.com/watch?v="+element
              yt = YouTube(source)
              stream = yt.streams.filter(subtype='mp4', res="480p").first()
              stream.download(output_path=dest)
              num_dl_videos += 1
              print "%d Video Dowloaded\n" %(num_dl_videos)
          
        


if __name__ == "__main__":
    youtube_download('sushi', 100, '720p', './videos')
  

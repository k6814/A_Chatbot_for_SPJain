#!/usr/bin/env python
# coding: utf-8

# In[1]:


a = []
import TeleBot
print("imported_telebot")

# In[ ]:


import json
import requests
import time


TOKEN = ''
URL = "https://api.telegram.org/bot{}/".format(TOKEN)
                

def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js


def get_updates(offset=None):
    url = URL + "getUpdates"
    if offset:
        url += "?offset={}".format(offset)
    js = get_json_from_url(url)
    for i in enumerate(js['result']):
        if 'text' not in i[1]['message']:
            del js['result'][i[0]]
    return js

def dictIDs():
    for i in get_updates()['result']:
        ID = i['message']['from']['id']
        TeleBot.chatTracker[ID] = {}
        TeleBot.chatTracker[ID]['c'] = ''
        TeleBot.chatTracker[ID]['prev'] = ''
        TeleBot.chatTracker[ID]['rep'] = False
        TeleBot.chatTracker[ID]['Help'] = False
        TeleBot.chatTracker[ID]['repeat'] = False
        TeleBot.chatTracker[ID]['progQ'] = False


def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)

#q, chat, c, prev, rep, Help, repeat, progQ
def echo_all(updates):
    for i in get_updates()['result']:
        if i['message']['from']['id'] not in TeleBot.chatTracker:
            ID = i['message']['from']['id']
            TeleBot.chatTracker[ID] = {}
            TeleBot.chatTracker[ID]['c'] = ''
            TeleBot.chatTracker[ID]['prev'] = ''
            TeleBot.chatTracker[ID]['rep'] = False
            TeleBot.chatTracker[ID]['Help'] = False
            TeleBot.chatTracker[ID]['repeat'] = False
            TeleBot.chatTracker[ID]['progQ'] = False
    for update in updates["result"]:
        text = update["message"]["text"]
        chat = update["message"]["chat"]["id"]
        if text != '/start':
            send_message(TeleBot.spjBot(text, chat, TeleBot.chatTracker[chat]['c'], \
                                        TeleBot.chatTracker[chat]['prev'], \
                                        TeleBot.chatTracker[chat]['rep'], \
                                        TeleBot.chatTracker[chat]['Help'], \
                                        TeleBot.chatTracker[chat]['repeat'], \
                                        TeleBot.chatTracker[chat]['progQ']), chat)


def get_last_chat_id_and_text(updates):
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)


def send_message(text, chat_id):
    #text = urllib.parse.quote_plus(text)
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    get_url(url)


def main():
    last_update_id = None
    dictIDs()
    while True:
        for i in get_updates()['result']:
            if i['message']['from']['id'] not in TeleBot.chatTracker:
                a.append(i)
                print(i)
                ID = i['message']['from']['id']
                TeleBot.chatTracker[ID] = {}
                TeleBot.chatTracker[ID]['c'] = ''
                TeleBot.chatTracker[ID]['prev'] = ''
                TeleBot.chatTracker[ID]['rep'] = False
                TeleBot.chatTracker[ID]['Help'] = False
                TeleBot.chatTracker[ID]['repeat'] = False
                TeleBot.chatTracker[ID]['progQ'] = False
        updates = get_updates(last_update_id)
        if len(updates["result"]) > 0:
            last_update_id = get_last_update_id(updates) + 1
            echo_all(updates)
        time.sleep(0.5)

if __name__ == '__main__':
    main()


# In[ ]:



#temp

# In[3]:


TeleBot.chatTracker


# In[4]:


a


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:





# In[ ]:





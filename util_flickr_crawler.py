import datetime
import json
import os
import pickle
from queue import Queue
from threading import Thread
from urllib import request
from urllib.parse import urlencode

import redis

import config


class Counter:
    def __init__(self):
        self.__date = datetime.date.today()

    def pre(self):
        self.__date -= datetime.timedelta(1)

    def next(self):
        self.__date += datetime.timedelta(1)

    def get_date(self):
        return self.__date.isoformat()


class RedisCounter:
    def __init__(self, name, namespace='counter', **redis_kwargs):
        self.__db = redis.Redis(**redis_kwargs)
        self.key = '%s:%s' % (namespace, name)
        self.__date = self.__db.get(self.key)
        if self.__date is None:
            self.__date = datetime.date.today()
        else:
            self.__date = pickle.loads(self.__date)

    def pre(self):
        self.__date -= datetime.timedelta(1)
        self.__update_to_redis()

    def next(self):
        self.__date += datetime.timedelta(1)
        self.__update_to_redis()

    def __update_to_redis(self):
        self.__db.set(self.key, pickle.dumps(self.__date))

    def get_date(self):
        return self.__date.isoformat()


class RedisQueue:
    """Simple Queue with Redis Backend"""

    def __init__(self, name, namespace='queue', **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""
        self.__db = redis.Redis(**redis_kwargs)
        self.key = '%s:%s' % (namespace, name)

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        self.__db.rpush(self.key, pickle.dumps(item))

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        if block:
            item = self.__db.blpop(self.key, timeout=timeout)
        else:
            item = self.__db.lpop(self.key)

        if item:
            item = item[1]
        return pickle.loads(item)

    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)


class ImageFinder(Thread):
    def __init__(self, date_start=None, database=None):
        super(ImageFinder, self).__init__()
        self.base_url = "https://api.flickr.com/services/rest/"
        self.__date = date_start
        self.__db = database

    def run(self):
        while True:
            if self.__db.qsize() < 10:
                try:
                    self.__parse_page(self.__date.get_date())
                except Exception as e:
                    pass
                self.__date.pre()

    def __parse_page(self, day):
        for ith_page in range(max(1, config.crawler['pages_per_day'])):
            header, param = self.__class__.__param_gen(day, page=ith_page + 1)
            print(self.base_url + "?" + param.decode('utf-8'))
            data = json.loads(request.urlopen(self.base_url, param, timeout=5).read().decode('utf-8'))
            if 'ok' == data['stat']:
                for item in data["photos"]["photo"]:
                    image_url = "https://farm{farm}.staticflickr.com/{server}/{id}_{secret}_b.jpg".format(
                        farm=item['farm'],
                        server=item['server'],
                        id=item['id'],
                        secret=item['secret']
                    )
                    self.__db.put({"id": item['id'], "url": image_url})
            elif 'fail' == data['stat']:
                print('Code:', data['code'], 'Message:', data['message'])
            else:
                print('Unknow Error')

    @classmethod
    def __param_gen(cls, day, **kwargs):
        headers = {
            "Connection": "keep-alive",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, sdch",
            "DNT": "1",
            "Accept-Language": "zh-CN,zh;q=0.8,en;q=0.6,zh-TW;q=0.4"
        }
        request_parameter = {
            "format": "json",
            "method": "flickr.interestingness.getList",
            "api_key": config.crawler['api_key'],
            "nojsoncallback": "1",
            "per_page": 10
        }
        request_parameter["date"] = day
        for (key, value) in kwargs.items():
            request_parameter[key] = value
        return headers, urlencode(request_parameter).encode()


class DataSaver(Thread):
    def __init__(self, database=None, base_dir=""):
        super(DataSaver, self).__init__()
        self.base_dir = base_dir
        self.__db = database

    def run(self):
        while True:
            record = self.__db.get()
            try:
                request.urlretrieve(record["url"], os.path.join(config.PATH_CRAW, record["id"] + ".jpg"))
            except Exception as e:
                print(e)


if '__main__' == __name__:
    if config.USE_REDIS:
        counter = RedisCounter('flickr_counter')
        queue = RedisQueue('flickr_queue')
    else:
        counter = Counter()
        queue = Queue()
    image_finder = ImageFinder(counter, queue)
    image_finder.start()
    for i in range(config.crawler['savers']):
        ds = DataSaver(queue)
        ds.start()

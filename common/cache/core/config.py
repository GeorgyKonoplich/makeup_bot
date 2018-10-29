import json
import os
from datetime import timedelta


class RootCustomError(Exception):
    def __init__(self, message):
        super(Exception, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.get_message())

    def get_message(self):
        return '{}!'.format(self.message)


class Singleton(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            try:
                cls._instance = cls.__new__(cls, *args, **kwargs)
                cls.__init__(cls, *args, **kwargs)
            except Exception as ex:
                raise RootCustomError('Config load error. Object: {}. Error: {}'.format(cls, ex))
        elif not isinstance(cls._instance, cls):
            raise RootCustomError('Instance wrong type: {}. Correct type: {}'.format(type(cls._instance), cls))

        return cls._instance


class RedisConfig():
    def __init__(self, base_dir):
        path = base_dir + '/configs/redis_params.json'
        with open(path, 'r') as f:
            redis_config = json.load(f)

        self.__host = redis_config['host']
        self.__port = redis_config['port']
        self.__db = redis_config['db']
        self.__default_ttl = int(timedelta(days=redis_config['default_ttl_in_days']).total_seconds())
        self.__password = redis_config.get('password')

    @property
    def host(self):
        return self.__host

    @property
    def port(self):
        return self.__port

    @property
    def db(self):
        return self.__db

    @property
    def default_ttl(self):
        return self.__default_ttl

    @property
    def password(self) :
        return self.__password

import pickle

import redis

from common.cache.core.config import RedisConfig


def get_data(key, base_dir):
    rc = RedisConfig(base_dir)
    cache = redis.StrictRedis(
        host=rc.host,
        port=rc.port,
        db=rc.db,
        password=rc.password
    )
    value = cache.get(name=key)

    if value:
        value = pickle.loads(value)

    return value


def set_data(data, base_dir):
    # verify
    if not type(data) is dict:
        return False
    data_keys = list(data.keys())
    rc = RedisConfig(base_dir)
    cache = redis.Redis(host=rc.host, port=rc.port, db=rc.db, password=rc.password)
    pipe = cache.pipeline()

    [pipe.set(name=key, value=pickle.dumps(value), ex=rc.default_ttl) for key, value in data.items()]
    exec_result = pipe.execute()

    if None in exec_result:
        result = False
    else:
        result = True

    return result


def delete_data(key, base_dir):
    rc = RedisConfig(base_dir)
    cache = redis.StrictRedis(host=rc.host, port=rc.port, db=rc.db, password=rc.password)
    result = cache.delete(key)

    return result

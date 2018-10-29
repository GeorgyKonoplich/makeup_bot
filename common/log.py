import json
import logging.config
import os
import random
from datetime import datetime
from string import ascii_lowercase

BASE_LOG_DIR = '/home/georgy/logs/'


def configure_loggers_with_file(base_dir):
    print(base_dir)
    logging_param_file = os.path.join(base_dir, 'configs/logging_params.json')
    with open(logging_param_file, 'r') as f:
        logger_config = json.load(f)

        # Log file path
        log_file_name = generate_unique_file_name(
            directory_name=BASE_LOG_DIR,
            file_name_prefix=datetime.now().strftime('%Y%m%d-%H%M'),
            name_format='{0}_{1}.log'
        )
        logger_config['handlers']['file']['filename'] = log_file_name

        logging.config.dictConfig(logger_config)


def generate_unique_file_name(directory_name, file_name_prefix, name_format):
    while True:
        generated_string = ''.join(random.choice(ascii_lowercase) for i in range(12))
        file_name = name_format.format(file_name_prefix, generated_string)
        file_path = os.path.join(directory_name, file_name)

        if not os.path.exists(file_path):
            return file_path

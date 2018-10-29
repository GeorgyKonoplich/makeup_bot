#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging

import flask
import telebot
import argparse
import tensorflow as tf

tf.set_random_seed(19)
from model import cyclegan, cyclegan1
from bot_utils import *

from common.cache.service import set_data, get_data

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')

parser.add_argument('--load_size', dest='load_size', type=int, default=128, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=128, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True,
                    help='generation network using reidule block')


args = parser.parse_args()

model_graph = tf.Graph()
with model_graph.as_default():
    model = cyclegan1('', args)

adv_graph = tf.Graph()
with adv_graph.as_default():
    model_eye = cyclegan('eyeeyeeye', args)

adv_sess = tf.Session(graph=adv_graph)
sess = tf.Session(graph=model_graph)
model.sess = sess
model_eye.sess = adv_sess

API_TOKEN = '743645761:AAE-s66t5tTC6k7ncqV8O--RRCFED8Cj050'

WEBHOOK_HOST = 'makeup.neurohive.net'
APP_PORT = 6543
WEBHOOK_LISTEN = '0.0.0.0'  # In some VPS you may need to put here the IP addr
SOURCE_DIR_CODE = '/home/georgy.konoplich/workspace/cycle_gan/'
WEBHOOK_URL_BASE = "https://%s" % (WEBHOOK_HOST)
WEBHOOK_URL_PATH = "/%s/" % (API_TOKEN)
mouth_weights = '/checkpoint/mouth2mouth_128_20_epochs'
eye_weights = '/checkpoint/eye2eye_128'
reference_mouth_list = []
reference_eye_list = []

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

bot = telebot.TeleBot(API_TOKEN)

app = flask.Flask(__name__)

with sess.as_default():
    with model_graph.as_default():
        tf.global_variables_initializer().run()
        model_saver = tf.train.Saver(tf.global_variables())
        model_ckpt = tf.train.get_checkpoint_state(SOURCE_DIR_CODE + mouth_weights)
        model_saver.restore(sess, model_ckpt.model_checkpoint_path)

with adv_sess.as_default():
    with adv_graph.as_default():
        tf.global_variables_initializer().run()
        adv_saver = tf.train.Saver(tf.global_variables())
        adv_ckpt = tf.train.get_checkpoint_state(SOURCE_DIR_CODE + eye_weights)
        adv_saver.restore(adv_sess, adv_ckpt.model_checkpoint_path)


def get_makeup_mouth(image, reference_mouth, intensity):
    image_mouth, mask = get_mouth(image)
    new_mouth = model.test_bot(args, image_mouth, SOURCE_DIR_CODE + reference_mouth, intensity)
    result_image = clone(mask, image, new_mouth)
    return result_image


def get_makeup_eyes(image, reference_eye, intensity):
    image_eye, mask = get_left_eye(image)

    new_eye = model_eye.test_bot(args, cv2.cvtColor(image_eye.astype(np.uint8), cv2.COLOR_BGR2RGB),
                             SOURCE_DIR_CODE + reference_eye, intensity)

    result_image = clone(mask, image, cv2.cvtColor(new_eye, cv2.COLOR_BGR2RGB))
    image_eye, mask = get_right_eye(result_image)
    new_eye = model_eye.test_bot(args, cv2.cvtColor(image_eye.astype(np.uint8), cv2.COLOR_BGR2RGB),
                             SOURCE_DIR_CODE + reference_eye, intensity)
    result_image = clone(mask, result_image, np.flip(cv2.cvtColor(new_eye, cv2.COLOR_BGR2RGB), axis=1))
    return result_image


def get_makeup(image, file_id, intensity, reference_mouth, reference_eye):
    output = get_makeup_mouth(image, reference_mouth, intensity)
    output = get_makeup_eyes(output, reference_eye, intensity)
    cv2.imwrite(SOURCE_DIR_CODE + '/bots_images/res' + str(file_id) + '.jpg', output)
    return SOURCE_DIR_CODE + '/bots_images/res' + str(file_id) + '.jpg'


# Empty webserver index, return nothing, just http 200
@app.route('/', methods=['GET', 'HEAD'])
def index():
    return ''


# Process webhook calls
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    if flask.request.headers.get('content-type') == 'application/json':
        json_string = flask.request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        flask.abort(403)


# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message,
                 "Hi there, I am MakeUpBot. Just upload your photo and see the magic =)\n")


# Handle document messages
@bot.message_handler(func=lambda message: True, content_types=['document'])
def echo_message(message):
    cid = message.chat.id
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    img = cv2.imdecode(np.frombuffer(downloaded_file, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    cv2.imwrite(SOURCE_DIR_CODE + '/bots_images/orig' + str(message.document.file_id) + '.jpg', img)
    set_data({cid: SOURCE_DIR_CODE + '/bots_images/orig' + str(message.document.file_id) + '.jpg'}, app.root_path)
    bot.reply_to(message, "Type makeup intensity number, 0 - without makeup, 100 - full, 200 - 2x.\n")

# Handle photo messages
@bot.message_handler(func=lambda message: True, content_types=['photo'])
def echo_message(message):
    cid = message.chat.id
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    img = cv2.imdecode(np.frombuffer(downloaded_file, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    cv2.imwrite(SOURCE_DIR_CODE + '/bots_images/orig' + str(message.photo[-1].file_id) + '.jpg', img)
    set_data({cid: SOURCE_DIR_CODE + '/bots_images/orig' + str(message.photo[-1].file_id) + '.jpg'}, app.root_path)
    bot.reply_to(message, "Type makeup intensity number, 0 - without makeup, 100 - full, 200 - 2x.\n")

# Handle text messages
@bot.message_handler(func=lambda message: True, content_types=['text'])
def echo_message(message):
    cid = message.chat.id
    #print(message.message_id, flush=True)
    path_to_image = get_data(cid, app.root_path)
    if path_to_image is None:
        bot.reply_to(message, "Upload photo first\n")
    else:
        if not message.text.isdigit():
            bot.reply_to(message, "Type only number\n")
        else:
            img = cv2.imread(path_to_image)
            #try:

            result_image_path = get_makeup(img, message.message_id, int(message.text) % 201,
                                               '/datasets/mouth2mouth/mouth_77.jpg',
                                               '/datasets/eye2eye/lefteye_302.jpg')
            bot.send_photo(cid, photo=open(result_image_path, 'rb'))
            #except Exception:
            #    bot.send_photo(cid, photo=open(path_to_image, 'rb'))

# Start flask server
app.run(host=WEBHOOK_LISTEN,
        port=APP_PORT,
        debug=True)


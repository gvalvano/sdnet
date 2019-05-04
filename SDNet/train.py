"""
For the training starting from zero, without fine tuning
"""
import tensorflow as tf
tf.random.set_random_seed(1234)
from model import Model
from idas.logger import telegram_notifier
import logging

N_EPOCHS = 60
NOTIFY = True

# telegram bot ---
TELEGRAM_TOKEN = '696365945:AAEZgDVuEkc7SF1iqbT0zR2YolbCvUwdfT4'  # token-id
TELEGRAM_CHAT_ID = '171620634'  # chat-id
# ----------------

if __name__ == '__main__':
    print('\n' + '-'*10)
    model = Model()
    model.build()

    if NOTIFY:
        try:
            model.train(n_epochs=N_EPOCHS)
            tel_message = 'Training finished.'
        except Exception as exc:
            print(exc)
            tel_message = str(exc) + '\nAn error arised. Check your code!'

        # - - - - -
        # Telegram notification:
        msg = "Automatic message from server 'imt_lucca'\n{separator}\n" \
              "<b>RUN_ID: </b>\n<pre> {run_id} </pre>\n" \
              "<b>MESSAGE: </b>\n<pre> {message} </pre>".format(run_id=model.run_id,
                                                                message=tel_message,
                                                                separator=' -' * 10)
        telegram_notifier.basic_notifier(logger_name='training_notifier',
                                         token_id=TELEGRAM_TOKEN,
                                         chat_id=TELEGRAM_CHAT_ID,
                                         message=msg,
                                         level=logging.INFO)
    else:
        model.train(n_epochs=N_EPOCHS)

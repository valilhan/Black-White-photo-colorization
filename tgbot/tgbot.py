import logging

from aiogram import Bot, Dispatcher, executor, types

from main import *


API_TOKEN = '6495435192:AAH8Ilt2O5j3gzF5AiQNZC-gFft38MasxnQ'


logging.basicConfig(level=logging.INFO)


bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load('res34-unet_g.pt', map_location=device))
model = MainModel(net_G=net_G)
model.load_state_dict(torch.load("res34-unet_model.pt", map_location=device))


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Hi! Send me a photo, and I'll colorize it.")


@dp.message_handler(content_types=['photo'])
async def colorize_and_send_photo(message: types.Message):
    await message.photo[-1].download('input_image.jpg')

    # Process and colorize the image using your model
    input_image_path = 'input_image.jpg'
    output_image_path = 'colorized_image.jpg'
    colorize_image(input_image_path, model, device, output_image_path)

    # Send back the colorized image
    with open(output_image_path, 'rb') as photo:
        await message.reply_photo(photo)


if __name__ == '__main__':
    # Start the bot
    executor.start_polling(dp, skip_updates=True)

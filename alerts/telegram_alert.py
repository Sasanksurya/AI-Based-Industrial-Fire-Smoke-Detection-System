import requests

import requests

BOT_TOKEN = "8826492709:AAGY0pZIvmuzi7rlHBAXjJ9p8zkJwTXQe3w"
CHAT_ID = "6827245492"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=payload)

        if response.status_code == 200:
            print("Telegram alert sent successfully")
        else:
            print("Telegram failed:", response.text)

    except Exception as e:
        print("Telegram error:", e)
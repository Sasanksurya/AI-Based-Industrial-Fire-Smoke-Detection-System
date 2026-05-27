import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject, message):
    sender_email = "shashanksurya24@gmail.com"
    sender_password = "jqgb odgc qwdb ywkt"
    receiver_email = "shashanksurya24@gmail.com"

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email alert sent successfully")

    except Exception as e:
        print("Email failed:", e)
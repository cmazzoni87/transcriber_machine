import requests
import os
from tools import DATA_DIR


def send_email_with_mailgun(subject, body, recipient_email, file_path=None):
    try:
        files = []
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as attachment:
                files = [("attachment", (os.path.basename(file_path), attachment.read()))]
                # Always attach transcript.txt if it exists

        transcript_file = DATA_DIR / "transcript.txt"
        if os.path.exists(transcript_file):
            with open(transcript_file, "rb") as transcript:
                files.append(("attachment", (os.path.basename(transcript_file), transcript.read())))

        response = requests.post(
            f"https://api.mailgun.net/v3/{os.getenv('MAILGUN_DOMAIN')}/messages",
            auth=("api", os.getenv('MAILGUN_API_KEY')),
            data={"from": os.getenv('EMAIL_ADDRESS'),
                  "to": [recipient_email],
                  "subject": subject,
                  "text": body},
            files=files if files else None

        )

        return response
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")
        return None
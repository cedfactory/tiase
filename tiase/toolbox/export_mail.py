# references :
# https://docs.python.org/3/library/email.examples.html
# https://realpython.com/python-send-email/
# https://www.quennec.fr/trucs-astuces/langages/python/python-envoyer-un-mail-tout-simplement
import smtplib, ssl
from email.message import EmailMessage
import mimetypes # For guessing MIME type based on file name extension
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_mail(password, message, attachments=None):
    print("[send_mail]")
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    sender_email = "cedfactory33@gmail.com"
    receiver_email = "cedfactory33@gmail.com"
    #password = input("Type your password and press enter: ")

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    server = smtplib.SMTP(smtp_server, port)
    try:
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login(sender_email, password)

        if attachments == None:
            server.sendmail(sender_email, receiver_email, message)
        else:
            msg = EmailMessage()
            msg['Subject'] = "Subject"
            msg['From'] = sender_email
            msg['To'] = receiver_email
            for filepath in attachments:
                ctype, encoding = mimetypes.guess_type(filepath)
                if ctype is None or encoding is not None:
                    ctype = 'application/octet-stream'
                maintype, subtype = ctype.split('/', 1)
                with open(filepath, 'rb') as fp:
                    msg.add_attachment(fp.read(),
                               maintype=maintype,
                               subtype=subtype,
                               filename=filepath)

            msg.attach(MIMEText(message, "plain"))
            server.send_message(msg)
        
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit() 

#! /usr/bin/env python

import glob
import os

# for regular expression
import re 

# Import smtplib for the actual sending function
import smtplib

# import MIME modules for message creation
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


p = re.compile('([a-zA-Z]{3}[0-9]{4})')

# On rMBP
#os.chdir("/Users/josh/Documents/Classes/MCHE485 - Spring 2015/Homework/HW 1 Submissions")

# On iMac
#os.chdir("/Users/josh/Documents/Classes/MCHE485 - Fall 2014/Mid-Term 2/Scans")

path = os.getcwd()

if 'MCHE485' in path:
    print('Returning MCHE485 Homework')
    course = 'MCHE485'
elif 'MCHE201' in path:
    print('Returning MCHE201 Homework')
    course = 'MCHE201'
elif 'MCHE513' in path:
    print('Returning MCHE513 Homework')
    course = 'MCHE513'
else:
    raise ValueError('\n\nAre you sure about this directory?\n')

for pdffile in glob.glob("*.pdf"):
    CLID_match = p.match(pdffile)
    to_address = CLID_match.group() + '@louisiana.edu'
    
    print('Sending to: ' + to_address)
    
    msg = MIMEMultipart()
    msg['Subject'] = 'Graded {} Homework'.format(course)
    msg['From'] = 'your_email_address'
    msg['To'] = to_address
    
    # Add the body text
    body = MIMEText('Attached is your graded {} homework. Comments may not show up on smartphones or tablets, but should on PCs or Macs. \n\n\
Thanks,\n\
JV\n\n\
-- \n\
Joshua Vaughan\n\
Assistant Professor\n\
Dept. of Mechanical Engineering\n\
Univ. of Louisiana at Lafayette\n\n'.format(course))
    msg.attach(body)
    
    # Add the attachement
    part = MIMEApplication(open(pdffile,"rb").read())
    part.add_header('Content-Disposition', 'attachment', filename=pdffile)
    msg.attach(part)
    
    # Send the message via the UL SMTP server
    mailer = smtplib.SMTP_SSL('mailer.zimbra.louisiana.edu')
#     mailer.starttls()
    mailer.login('your_email_address','your_email_password')
    mailer.sendmail(msg['From'], msg['To'] , msg.as_string())
    mailer.quit()
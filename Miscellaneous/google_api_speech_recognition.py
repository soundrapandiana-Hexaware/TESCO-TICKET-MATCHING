# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:33:47 2019

@author: 41122
"""

import speech_recognition as sr 
r = sr.Recognizer()

ricoh_data = sr.AudioFile('Issue1.wav')

def speech_to_text_recorded_data():
	with ricoh_data as source:
		#r.adjust_for_ambient_noise(source)
		audio = r.record(source)
	text_data = r.recognize_google(audio,language = "en-US")      
	print(text_data)

speech_to_text_recorded_data()
	
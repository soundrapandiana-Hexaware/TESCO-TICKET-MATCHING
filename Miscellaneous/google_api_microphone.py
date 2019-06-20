# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:55:22 2019

@author: Vithya
"""

import speech_recognition as sr

r = sr.Recognizer()

def speech_to_text_using_microphone():
	with sr.Microphone() as source:
		print("Say something!")
		audio = r.listen(source)
	# Speech recognition using Google Speech Recognition
	try:
		# for testing purposes, we're just using the default API key
		# to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
		# instead of `r.recognize_google(audio)`
		print("You said: " + r.recognize_google(audio))
	except sr.UnknownValueError:
		print("Google Speech Recognition could not understand audio")
	except sr.RequestError as e:
		print("Could not request results from Google Speech Recognition service; {0}".format(e))
	
speech_to_text_using_microphone()	
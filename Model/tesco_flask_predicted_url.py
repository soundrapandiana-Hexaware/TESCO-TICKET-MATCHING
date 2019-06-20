# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:05:38 2019

@author: Vithya
"""

import json
import os
import traceback
from flask_restful import Resource, Api, request
from tesco_ml_predictions import ExtractTypes
from tokenization import tokenize_only
from flask import Flask
app = Flask(__name__)
api = Api(app)
extractTypes  = ExtractTypes()


class ClassifyTescoCategory(Resource):
    def __init__(self):
        #self.parser = reqparse.RequestParser()
        pass
    def get(self):
        return {'response': 'Please use POST. GET is not supported'}
    
    def post(self):
        result_dict = {}

        try:
            #print("request.data: ", request.data)
            file_path = json.loads(request.data.decode("UTF-8"))
            print("request.data....: ", file_path)
            itext = file_path['Ticket_Subject']        
            result_dict['Tesco'] = extractTypes.get_text_category(itext)
        except:
            print(traceback.format_exc()) 
            result_dict['Tesco'] = 'Document type updating failed'
        return json.loads(json.dumps(result_dict))
    
api.add_resource(ClassifyTescoCategory, '/api/v1.0/subcategory')


if __name__ == "__main__":
	app.run(host='127.0.0.1', port=5004, debug=True)
#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
"""
Skeleton code showing how to load and run the TensorFlow Lite export package from Lobe.
"""

import argparse
import json
import os
import time

import io
import picamera
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

from datetime import datetime


def get_model_and_sig(model_dir):
    """Method to get name of model file. Assumes model is in the parent directory for script."""
    with open(os.path.join(model_dir, "signature.json"), "r") as f:
        signature = json.load(f)
        #print(signature.get("filename"))
    model_file = model_dir + "/" + signature.get("filename")
    #print(model_file)
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file does not exist")
    return model_file, signature


def load_model(model_file):
    """Load the model from path to model file"""
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    return interpreter


def get_prediction(image, interpreter, signature):
    """
    Predict with the TFLite interpreter!
    """
    # Combine the information about the inputs and outputs from the signature.json file with the Interpreter runtime
    signature_inputs = signature.get("inputs")
    input_details = {detail.get("name"): detail for detail in interpreter.get_input_details()}
    model_inputs = {key: {**sig, **input_details.get(sig.get("name"))} for key, sig in signature_inputs.items()}
    signature_outputs = signature.get("outputs")
    output_details = {detail.get("name"): detail for detail in interpreter.get_output_details()}
    model_outputs = {key: {**sig, **output_details.get(sig.get("name"))} for key, sig in signature_outputs.items()}

    if "Image" not in model_inputs:
        raise ValueError("Tensorflow Lite model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")

    # process image to be compatible with the model
    input_data = process_image(image, model_inputs.get("Image").get("shape"))

    # set the input to run
    interpreter.set_tensor(model_inputs.get("Image").get("index"), input_data)
    interpreter.invoke()

    # grab our desired outputs from the interpreter!
    # un-batch since we ran an image with batch size of 1, and convert to normal python types with tolist()
    outputs = {key: interpreter.get_tensor(value.get("index")).tolist()[0] for key, value in model_outputs.items()}
    # postprocessing! convert any byte strings to normal strings with .decode()
    for key, val in outputs.items():
        if isinstance(val, bytes):
            outputs[key] = val.decode()

    return outputs


def process_image(image, input_shape):
    """
    Given a PIL Image, center square crop and resize to fit the expected model input, and convert from [0,255] to [0,1] values.
    """
    width, height = image.size
    # ensure image type is compatible with model and convert if not
    if image.mode != "RGB":
        image = image.convert("RGB")
    # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
    if width != height:
        square_size = min(width, height)
        left = (width - square_size) / 2
        top = (height - square_size) / 2
        right = (width + square_size) / 2
        bottom = (height + square_size) / 2
        # Crop the center of the image
        image = image.crop((left, top, right, bottom))
    # now the image is square, resize it to be the right shape for the model input
    input_width, input_height = input_shape[1:3]
    if image.width != input_width or image.height != input_height:
        image = image.resize((input_width, input_height))

    # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
    image = np.asarray(image) / 255.0
    # format input as model expects
    return image.reshape(input_shape).astype(np.float32)


def main(image, model_dir):
    """
    Load the model and signature files, start the TF Lite interpreter, and run prediction on the image.

    Output prediction will be a dictionary with the same keys as the outputs in the signature.json file.
    """
    model_file, signature = get_model_and_sig(model_dir)
    interpreter = load_model(model_file)
    prediction = get_prediction(image, interpreter, signature)
    # get list of confidences from prediction
    confidences = list(prediction.values())[0]
    # get the label name for the predicted class
    labels = signature.get("classes").get("Label")
    max_confidence = max(confidences)
    prediction["Prediction"] = labels[confidences.index(max_confidence)]
    return prediction

def chairStateModel(image):
    """
    This function uses the Presence_TFLite model to determine whether the user is in their chair
    """
    # Assume model is in the parent directory for this file
    model_dir = os.getcwd() + "/Presence_TFLite_2"
    #run the first model that detects presence in the chair
    prediction = main(image, model_dir)
    #extract the state - is there someone in the chair?
    sittingState = prediction["Prediction"]
    
    print("Prediction : " + prediction["Prediction"])
    print("Confidence : " + str(prediction["Confidences"]))
    
    return sittingState

def userStateModel(image, focusTime, phoneTime, nonFocusTime):
    """
    Checks to see if the user is focused or not and returns edited values for those
    """
    model_dir = os.getcwd() + "/UserState_TFLite"
    #run the first model that detects presence in the chair
    UserPrediction = main(image, model_dir)
    UserState = UserPrediction["Prediction"]
    UserStateConfidences = UserPrediction["Confidences"]
    print(UserState)
    print(UserStateConfidences)
    
    if UserState == "Focused":
        if UserStateConfidences[0] > 0.99:
            #this is a definitely focused state
            focusTime += 1
            return focusTime, phoneTime, nonFocusTime
        
        else:
            #this is a nonfocused, but not on phone state
            nonFocusTime += 1
            return focusTime, phoneTime, nonFocusTime
        
    elif UserState == "On_Phone":
        if UserStateConfidences[1] > 0.99:
            #definite phone state
            phoneTime += 1
            nonFocusTime += 1
            return focusTime, phoneTime, nonFocusTime
        
        else:
            #unfocused, but not definitely on phone
            nonFocusTime += 1
            return focusTime, phoneTime, nonFocusTime
        
    else:
        #catch possible errors
        return focusTime, phoneTime, nonFocusTime
            
    


def MLModels():
    #extract image with picamera
    camera = picamera.PiCamera(resolution=(1024, 768), framerate=10)
    #camera.vflip = True
    stream = io.BytesIO()
    
    #decalre values for the states
    sittingState = "" #starts with default
    focusTime = 0
    phoneTime = 0
    nonFocusTime = 0
    
    #decalre values for the timers
    s_time = time.time() - 16 #for the first loop
    prevSecond = 0
    
    while True:
        #every minute the values being calcualted from the model need to be pushed to a text
        #file so that they can be sent to the other Pi
        now = datetime.now()
        currentSecond = int(now.strftime("%S"))
        if currentSecond < prevSecond:
            #this means that a minute has passed and the second clock has returned to zero
            #hence push the values to the txt file
            totalTime = focusTime + nonFocusTime
            #calcualte the times as ratio for the full 60s
            focusTime = 60*(focusTime/totalTime)
            nonFocusTime = 60*(nonFocusTime/totalTime)
            phoneTime = 60*(phoneTime/totalTime)
            
            file1 = open("PiCamData.txt","w")
            #write this change to the text file
            print(sittingState + "," + str(round(focusTime,2)) + "," +  str(round(phoneTime,2))  + "," + str(round(nonFocusTime,2)))
            file1.write(sittingState + "," + str(focusTime) + "," +  str(phoneTime)  + "," + str(nonFocusTime))
            file1.close()
            focusTime = 0
            phoneTime = 0
            nonFocusTime = 0

        prevSecond = currentSecond

        #initate the camera and capture the picture
        camera.capture(stream, format='jpeg', use_video_port=True)
        stream.seek(0)
        #save image to stream
        image = Image.open(stream)
        
        #check whether use is in their chair every 30 seconds - this avoids clogging the programme with unnesscessary model runs
        if time.time() > s_time + 15:
            #run the model
            sittingState = chairStateModel(image)
            print(sittingState)            
            s_time = time.time()

        #If there is someone sitting in the chair, then apply the UserState detection model to them
        if sittingState == "Sitting_In_Chair":
            
            focusTime, phoneTime, nonFocusTime = userStateModel(image, focusTime, phoneTime, nonFocusTime)
            
        #reset the image stream
        stream.flush()
        stream = io.BytesIO()
        
if __name__ == "__main__":
    while True:
        try:
            MLModels()
        except:
            print("Model run failed, retrying...")

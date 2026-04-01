# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "SpeechRecognition"
# ]
# ///

import turtle
import speech_recognition as sr

wn = turtle.Screen()
wn.title("Voice Controlled Turtle")
wn.bgcolor("white")
wn.setup(width=600, height=600)
wn.tracer(0)

player = turtle.Turtle()
player.shape("turtle")
player.color("red")
player.penup()
speed = 20

def go_up():
    player.forward(speed)

def go_down():
    player.backward(speed)

def turn_left():
    player.left(45)

def turn_right():
    player.right(45)

recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_command():
    with mic as source:
        print("Escuchando...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio, language="es-ES")
        print("Dijiste:", command)
        return command.lower()
    except:
        return ""

while True:
    wn.update()
    
    command = listen_command()

    if "adelante" in command:
        go_up()
    elif "atras" in command:
        go_down()
    elif "izquierda" in command:
        turn_left()
    elif "derecha" in command:
        turn_right()
# door_checker.py
import random

def check_door_status():
    responses = [
        "都關了",
        "前門有關後門沒有關",
        "前門沒關後門有關",
        "都沒關"
    ]
    return random.choice(responses)

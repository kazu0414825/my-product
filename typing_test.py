import random
import time
from pynput import keyboard

sentences = [
    "Python is fun",
    "I love programming",
    "Data science is the future",
    "Machine learning changes the world",
    "Artificial intelligence is amazing"
]

def typing_test():
    target = random.choice(sentences)
    print("\n以下の文章をタイプしてください:")
    print(target)

    start_time = None
    input_buffer = ""

    def on_press(key):
        nonlocal start_time, input_buffer
        try:
            char = key.char
        except AttributeError:
            if key == keyboard.Key.enter:
                return False  
            return  
        if start_time is None:
            start_time = time.time() 
        input_buffer += char

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    t = time.time() - start_time
    total_chars = len(input_buffer)
    speed = total_chars/t
    
    t = time.time() - start_time
    
    correct_chars = sum(1 for a,b in zip(input_buffer,target) if a==b)
    accuracy = correct_chars/total_chars    
    
    print("\n結果:")
    print(f"かかった時間: {t:.2f} 秒")
    print(f"速度: {speed:.2f} 文字/秒")
    if total_chars == len(target):
        print(f"正確性: {accuracy:.2f} %")   
    else:
        print(f"正確性: 測定できませんでした")
              
typing_test()

import csv
from datetime import datetime

def save_result(speed, accuracy):
    with open("mentalwave_input_data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # 必要に応じて他の特徴量も追記する
        writer.writerow([datetime.now().strftime("%Y-%m-%d"), speed, accuracy])

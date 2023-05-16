##### Skeleton Code for Environment via Selenium ####

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

import matplotlib as plt

import io
import numpy as np
import time
#import wandb
from PIL import Image, ImageOps


class gd_environment:
    
    def __init__(self):
        
        self.browser_driver = get('https://scratch.mit.edu/projects/105500895/embed')
        self.browser_wait = WebDriverWait(self.browser_driver, timeout = 20)
        
        self.browser_wait.until(lambda d: d.find_element(By.CSS_SELECTOR, ".green-flag_green-flag_1kiAo"))
        green_flag = self.browser_driver.find_element(By.CSS_SELECTOR, ".green-flag_green-flag_1kiAo")
        time.sleep(2)
        green_flag.click()
        time.sleep(5)
        
        self.html_element = self.browser_driver.find_element(By.TAG_NAME, 'html')
        self.pause_down = webdriver.ActionChains(self.browser_driver).move_to_element(self.html_element).key_down("p")
        self.pause_up = webdriver.ActionChains(self.browser_driver).move_to_element(self.html_element).key_up("p")
        self.space_down = webdriver.ActionChains(self.browser_driver).move_to_element(self.html_element).key_down(Keys.SPACE)
        self.space_up = webdriver.ActionChains(self.browser_driver).move_to_element(self.html_element).key_down(Keys.SPACE)

        self.isPaused = False
        
    def get_score(self):
        
        self.monitor_element = self.browser_driver.find_element(By.CSS_SELECTOR, '.monitor-list_monitor-list-scaler_143tA .monitor_monitor-container_2J9gl[style="touch-action: none; transform: translate(0px, 0px); top: 6px; left: 5px;"] .monitor_value_3Yexa')
        return self.monitor_element.text
    
    def get_state():
       
        image_screenshot = self.browser_driver.find_element(By.CSS_SELECTOR, ".stage_stage_1fD7k canvas").screenshot_as_png
        image = Image.open(io.BytesIO(image_png))
        
        return np.asarray(image)
    
    def press_space(self):
        
        self.space_down.perform()
        self.space_up.perform()
    
    def press_pause(self):
        
        self.pause_down.perform()
        self.pause_up.perform()
     
    def pause_game(self):
        if not self.isPaused:
            self.press_pause()
            self.isPaused = True
            
    def unpause_game(self):
        if self.isPaused:
            self.press_pause()
            self.isPaused = False
            
        
    def end_gd_environment(self):
        self.browser_driver.quit()
        # SAVE AGENT HERE LATER



### Skeleton Code for Agent (Neural Network) ####

from os import path

import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
import random
#import wandb


class agent:
    
    def __init__(self, gamma = 0.95, eps_start = 0.9, eps_decay = 100):
        self.gamma = gamma # Discount factor
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        
        if(path.exists(self.get_model_name())):
            self.model = load_model(self.get_model_name())
        else:
            # Neural Network stuff here
            # Since we will be working with screenshots directly,
            # we will most likely want to construct a convolutional neural network
            # i.e. input = size of image output = 1 (either jump or don't jump)
            self.model = Sequential([
                layers.Conv2D(64, (3, 3), stride = 4, activation = 'relu', input_shape = (500, 500)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), stride = 2, activation = 'relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation = 'relu'),
                layers.Flatten(),
                layers.Dense(128, activation = 'relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation = 'sigmoid')
            ])

            self.model.compile(optimizer = 'adam',
                               loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                               metrics = ['accuracy'])
    
    def get_model_name(self):
        return 'geometry_dash_model.h5'
    
    def save_model(self):
        self.model.save(self.get_model_name())


### Everything Below This is Experimental Stuff (For Playing Around With) ###


#EXPERIMENTAL STUFF WITH SELENIUM

PATH = r"/Users/ilianamarrujo/computing16B/chromedriver"

driver_options = webdriver.ChromeOptions()
driver_options.add_argument('hide_browser')

driver = webdriver.Chrome(PATH, options = driver_options)

wait = WebDriverWait(driver, timeout = 20)


driver.get("https://scratch.mit.edu/projects/105500895/embed")
wait.until(lambda d: d.find_element(By.CSS_SELECTOR, ".green-flag_green-flag_1kiAo"))
green_flag = driver.find_element(By.CSS_SELECTOR, ".green-flag_green-flag_1kiAo")
time.sleep(2)
green_flag.click()
time.sleep(5)

image_screenshot = driver.find_element(By.CSS_SELECTOR, ".stage_stage_1fD7k canvas").screenshot_as_png
image = Image.open(io.BytesIO(image_screenshot))

image = image.resize((500, 500), Image.ANTIALIAS)
image = ImageOps.grayscale(image)
image.show()

image = np.asarray(image)
# print(image) 

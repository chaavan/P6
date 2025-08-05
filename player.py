from config import BOARD_SIZE, categories, image_size
from tensorflow.keras import models
import numpy as np
import tensorflow as tf

class TicTacToePlayer:
    def get_move(self, board_state):
        raise NotImplementedError()

class UserInputPlayer:
    def get_move(self, board_state):
        inp = input('Enter x y:')
        try:
            x, y = inp.split()
            x, y = int(x), int(y)
            return x, y
        except Exception:
            return None

import random

class RandomPlayer:
    def get_move(self, board_state):
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board_state[i][j] is None:
                    positions.append((i, j))
        return random.choice(positions)

from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2

class UserWebcamPlayer:
    def __init__(self):
        self.model = models.load_model('results/basic_model_50_epochs_timestamp_1754380339.keras')

    def _process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width, height = frame.shape
        size = min(width, height)
        pad = int((width-size)/2), int((height-size)/2)
        frame = frame[pad[0]:pad[0]+size, pad[1]:pad[1]+size]
        return frame

    def _access_webcam(self):
        import cv2
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
            frame = self._process_frame(frame)
        else:
            rval = False
        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            frame = self._process_frame(frame)
            key = cv2.waitKey(20)
            if key == 13: # exit on Enter
                break

        vc.release()
        cv2.destroyWindow("preview")
        return frame

    def _print_reference(self, row_or_col):
        print('reference:')
        for i, emotion in enumerate(categories):
            print('{} {} is {}.'.format(row_or_col, i, emotion))
    
    def _get_row_or_col_by_text(self):
        try:
            val = int(input())
            return val
        except Exception as e:
            print('Invalid position')
            return None
    
    def _get_row_or_col(self, is_row, board_state):
        while True:
            try:
                row_or_col_str = 'row' if is_row else 'col'
                self._print_reference(row_or_col_str)
                img = self._access_webcam()
                emotion = self._get_emotion(img)
                
                if type(emotion) is not int or emotion not in range(len(categories)):
                    print('Invalid emotion number {}. Please try again.'.format(emotion))
                    continue
                
                print('Emotion detected as {} ({} {}). Enter \'text\' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.'.format(categories[emotion], row_or_col_str, emotion))
                inp = input()
                if inp == 'text':
                    val = self._get_row_or_col_by_text()
                else:
                    val = emotion

                if val is None or val not in range(BOARD_SIZE):
                    print('Invalid input. Please enter 0, 1, or 2.')
                    continue

                # Validate the move against the board state
                if is_row:
                    temp_row = val
                    temp_col = None # We don't have the column yet, so can't fully validate
                else:
                    temp_row = None # We don't have the row yet
                    temp_col = val

                # This partial validation is tricky. The full validation happens in get_move.
                # For now, just ensure the value is within bounds.
                return val
            except Exception as e:
                print(f"Error accessing webcam or processing image: {e}. Please try again.")
                continue
    
    def _get_emotion(self, img) -> int:
        img_resized = cv2.resize(img, image_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_expanded = np.expand_dims(img_rgb, axis=0)
        prediction = self.model.predict(img_expanded)
        return int(np.argmax(prediction))
    
    def get_move(self, board_state):
        while True:
            row = self._get_row_or_col(True, board_state)
            col = self._get_row_or_col(False, board_state)

            if row is not None and col is not None:
                if board_state[row][col] is None:
                    return row, col
                else:
                    print(f"Position ({row}, {col}) is already taken. Please choose an empty spot.")
            else:
                print("Invalid row or column detected. Please try again.")

##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## timer
##

import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start

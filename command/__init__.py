import os
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
load_dotenv("E:/my_research/brain/config/.env")

VERSION = os.getenv("VERSION")
DIM = int(os.getenv("DIM"))

STORAGE = os.getenv("STORAGE_FOLDER")

# Một số biến dùng để lưu trữ khác
S_UNIT = "{}/units".format(STORAGE)
S_MODEL = "{}/models".format(STORAGE)
S_MANAGE = "{}/management".format(STORAGE)

def build_folder():
    if not os.path.exists(STORAGE):
        os.makedirs(STORAGE)
    if not os.path.exists(S_UNIT):
        os.makedirs(S_UNIT)
    if not os.path.exists(S_MODEL):
        os.makedirs(S_MODEL)
    if not os.path.exists(S_MANAGE):
        os.makedirs(S_MANAGE)

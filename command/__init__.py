import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
# path = Path("E:\\my_research\\brain\\.env")
# load_dotenv(path)
load_dotenv()

VERSION = os.getenv("VERSION")
DIM = int(os.getenv("DIM"))

STORAGE = os.getenv("STORAGE_FOLDER")
if not os.path.exists(STORAGE):
    os.makedirs(STORAGE)

# Một số biến dùng để lưu trữ khác
S_UNIT = "{}/units".format(STORAGE)
if not os.path.exists(S_UNIT):
    os.makedirs(S_UNIT)

S_MODEL = "{}/models".format(STORAGE)
if not os.path.exists(S_MODEL):
    os.makedirs(S_MODEL)

S_MANAGE = "{}/management".format(STORAGE)
if not os.path.exists(S_MANAGE):
    os.makedirs(S_MANAGE)
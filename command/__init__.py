import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
path = Path("../env")
load_dotenv(path)

VERSION = os.getenv("VERSION")
STORAGE = os.getenv("STORAGE")
DIM = int(os.getenv("DIM"))
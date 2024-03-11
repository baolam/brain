import os
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
load_dotenv()

VERSION = os.getenv("VERSION")
STORAGE = os.getenv("STORAGE")
DIM = int(os.getenv("DIM"))
import torch
from torch import nn as nn
import numpy as np
import torch.nn.functional as F
import random
import os
import cv2
from utils2 import *
import sys
import copy
from torch.utils.checkpoint import checkpoint
import pickle
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib import cm
from matplotlib.collections import LineCollection
import matplotlib as mpl
import datetime
import time
from dv import AedatFile
import sklearn.cluster as skc
from sklearn import metrics
import win32con
import win32clipboard as w
import keyboard
import requests
import zipfile
from PIL import ImageGrab
from aip import AipOcr
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import multiprocessing as mp
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon,QFont
import string
import sympy as sp
import scipy


def save(model,path):
	torch.save(model.state_dict(), path)


def load(model,path):
	model.load_state_dict(torch.load(path))
	return model
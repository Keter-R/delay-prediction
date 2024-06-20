import math
from cnocr import CnOcr
from PIL import Image
import cv2
import pytesseract
from thefuzz import process
from thefuzz import fuzz

def extract_stops_name_list_from_img(img_path):
    # in the image, the stops are in the form of a list with two columns: stop name and stop number
    # use ocr to extract the stops from the image
    # save stop names in a list and discard the stop numbers
    stops = []
    print('Extracting stops from image:', img_path)
    img = cv2.imread(img_path)
    texts = pytesseract.image_to_string(img, lang='eng')
    stops = texts.split('\n')
    # remove first line if content is 'Stop Name Stop Number'
    if stops[0] == 'Stop Name Stop Number':
        stops = stops[1:]
    # erase the first number from tail of each line and content after that number
    for i in range(len(stops)):
        stop = stops[i]
        for j in range(len(stop) - 1, -1, -1):
            if stop[j].isdigit():
                while j > 0 and stop[j - 1].isdigit():
                    j -= 1
                while j > 0 and not stop[j - 1].isalpha():
                    j -= 1
                stops[i] = stop[:j]
                break
    # remove blank lines
    stops = [stop for stop in stops if stop]
    return stops


def extract_stops_number_list_from_img(img_path):
    # in the image, the stops are in the form of a list with two columns: stop name and stop number
    # use ocr to extract the stops from the image
    # save stop numbers in a list and discard the stop names
    stops = []
    print('Extracting stops from image:', img_path)
    img = cv2.imread(img_path)
    # cut the image to the right half
    img = img[:, img.shape[1] // 2:]
    # convert img to binary image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # set pytesseract to ocr numbers only
    texts = pytesseract.image_to_string(img, config='digits', lang='eng')
    stops = texts.split('\n')
    # remove blank lines
    stops = [stop for stop in stops if stop]
    return stops


def match_station_with_stop(stops, location, confidence=80):
    # remove 'and' in the location to ''
    location = location.replace('and', '')
    # try to match the station name with a set of stop names
    res = process.extractOne(location, stops, scorer=fuzz.token_set_ratio)
    if res[1] >= confidence:
        return res[0]
    return None

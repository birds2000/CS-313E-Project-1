"""
# File: Project1.py
# Description: 
# Student Names: Katharine Smith and Isabel Svendsen
# Student EIDs: kbs2529, 
# Course Name: CS 313E
# Unique Number: 
# Date Created: 11/5/2025
# Date Last Modified: 
"""

import sys
import csv
from collections import namedtuple

Person = namedtuple("Person", ["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Salary"])

class DataCleaner:
    DegreeOrder = {
        "None": 0,
        "High School": 1,
        "Bachelor's": 2,
        "Bachelor's Degree": 2,
        "Master's Degree": 3,
        "PhD": 4
    }

def __init__(self, filename: str, col_map: dict = None):
    """Filename = path to CSV file, Col_Map = dict mapping keys to CSV column names."""
    self.filename = filename
    self.col_map = col_map or {
        "Age": "Age",
        "Gender": "Gender",
        "Education Level": "Education Level",
        "Job Title": "Job Title",
        "Years of Experience": "Years of Experience",
        "Salary": "Salary"
    }

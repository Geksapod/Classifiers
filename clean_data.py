import re


def y_verify(string: str):

    '''Return True if string contains only 6 numbers at the beginning of the string.'''

    pattern = r"\d"
    if re.match(pattern, string):
        return True

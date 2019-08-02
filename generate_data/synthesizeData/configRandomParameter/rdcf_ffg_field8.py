import random

dictConfig = {
    'field8': {
        'PRINTED_MIN_HEIGHT': 45,
        'PRINTED_MAX_HEIGHT': 64,
        'HW_MIN_HEIGHT': 57,
        'HW_MAX_HEIGHT': 70
    },
    'freeForm': {
        'PRINTED_MIN_HEIGHT': 22,
        'PRINTED_MAX_HEIGHT': 50,
        'HW_MIN_HEIGHT': 40,
        'HW_MAX_HEIGHT': 64
    },
    'field12': {
        'PRINTED_MIN_HEIGHT': 60,
        'PRINTED_MAX_HEIGHT': 70,
        'HW_MIN_HEIGHT': 57,
        'HW_MAX_HEIGHT': 70,
        'PT_MIN_SPACE': 10,
        'PT_MAX_SPACE': 30,
        'HW_MIN_SPACE': 10,
        'HW_MAX_SPACE': 25,
    }
}

dictChoose = dictConfig['field12']



def getConfigHeightPT():
    return random.randint(dictChoose['PRINTED_MIN_HEIGHT'], dictChoose['PRINTED_MAX_HEIGHT'])

def getConfigHeightHW():
    random.seed()
    return random.randint(dictChoose['HW_MIN_HEIGHT'], dictChoose['HW_MAX_HEIGHT'])

def getRandomSpaceCharPT():
    return random.randint(dictChoose['PT_MIN_SPACE'], dictChoose['PT_MAX_SPACE'])

def getRandomSpaceCharHW():
    return random.randint(dictChoose['HW_MIN_SPACE'], dictChoose['HW_MAX_SPACE'])

def getRandomHeightKata():
    random.seed()
    return random.randint(30,52)

def getKanakataConfig():
    return {
        'ONE_LINE_WIDTH': 760,
        'TWO_LINE_WIDTH': 1500,
        'FIX_ALL_LINE_HEIGHT': 80,
        'FIX_BOX_WIDTH': 58,
        'DELTA_CHAR_HEIGHT': 5,
        'MARGIN_BOTTOM_TOP': 10,
        'MAX_BOTTOM': 25, #maximum value of bottom of a char in box
    }
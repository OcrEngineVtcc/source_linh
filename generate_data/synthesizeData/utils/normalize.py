# Normalize and clean Japanese characters
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata

from main_source.utils.synthesizeData.utils.constants import DAKUTEN, HANDAKUTEN, CONVERSION_TABLE


KATA_SEPARATE_DIACRITICS = {
    'ガ': 'カ' + DAKUTEN,
    'ギ': 'キ' + DAKUTEN,
    'グ': 'ク' + DAKUTEN,
    'ゲ': 'ケ' + DAKUTEN,
    'ゴ': 'コ' + DAKUTEN,
    'ザ': 'サ' + DAKUTEN,
    'ジ': 'シ' + DAKUTEN,
    'ズ': 'ス' + DAKUTEN,
    'ゼ': 'セ' + DAKUTEN,
    'ゾ': 'ソ' + DAKUTEN,
    'ダ': 'タ' + DAKUTEN,
    'ヂ': 'チ' + DAKUTEN,
    'ヅ': 'ツ' + DAKUTEN,
    'デ': 'テ' + DAKUTEN,
    'ド': 'ト' + DAKUTEN,
    'バ': 'ハ' + DAKUTEN,
    'ビ': 'ヒ' + DAKUTEN,
    'ブ': 'フ' + DAKUTEN,
    'ベ': 'ヘ' + DAKUTEN,
    'ボ': 'ホ' + DAKUTEN,
    'ヴ': 'ウ' + DAKUTEN,
    'ヷ': 'ワ' + DAKUTEN,

    'パ': 'ハ' + HANDAKUTEN,
    'ピ': 'ヒ' + HANDAKUTEN,
    'プ': 'フ' + HANDAKUTEN,
    'ペ': 'ヘ' + HANDAKUTEN,
    'ポ': 'ホ' + HANDAKUTEN,
}

KATA_COMBINE_DIACRITICS = {
    'カ' + DAKUTEN: 'ガ',
    'キ' + DAKUTEN: 'ギ',
    'ク' + DAKUTEN: 'グ',
    'ケ' + DAKUTEN: 'ゲ',
    'コ' + DAKUTEN: 'ゴ',
    'サ' + DAKUTEN: 'ザ',
    'シ' + DAKUTEN: 'ジ',
    'ス' + DAKUTEN: 'ズ',
    'セ' + DAKUTEN: 'ゼ',
    'ソ' + DAKUTEN: 'ゾ',
    'タ' + DAKUTEN: 'ダ',
    'チ' + DAKUTEN: 'ヂ',
    'ツ' + DAKUTEN: 'ヅ',
    'テ' + DAKUTEN: 'デ',
    'ト' + DAKUTEN: 'ド',
    'ハ' + DAKUTEN: 'バ',
    'ヒ' + DAKUTEN: 'ビ',
    'フ' + DAKUTEN: 'ブ',
    'ヘ' + DAKUTEN: 'ベ',
    'ホ' + DAKUTEN: 'ボ',
    'ウ' + DAKUTEN: 'ヴ',
    'ワ' + DAKUTEN: 'ヷ',

    'ハ' + HANDAKUTEN: 'パ',
    'ヒ' + HANDAKUTEN: 'ピ',
    'フ' + HANDAKUTEN: 'プ',
    'ヘ' + HANDAKUTEN: 'ペ',
    'ホ' + HANDAKUTEN: 'ポ'
}

HIRA_SEPARATE_DIACRITICS = {
    'が': 'か' + DAKUTEN,
    'ぎ': 'き' + DAKUTEN,
    'ぐ': 'く' + DAKUTEN,
    'げ': 'け' + DAKUTEN,
    'ご': 'こ' + DAKUTEN,
    'ざ': 'さ' + DAKUTEN,
    'じ': 'し' + DAKUTEN,
    'ず': 'す' + DAKUTEN,
    'ぜ': 'せ' + DAKUTEN,
    'ぞ': 'そ' + DAKUTEN,
    'だ': 'た' + DAKUTEN,
    'ぢ': 'ち' + DAKUTEN,
    'づ': 'つ' + DAKUTEN,
    'で': 'て' + DAKUTEN,
    'ど': 'と' + DAKUTEN,
    'ば': 'は' + DAKUTEN,
    'び': 'ひ' + DAKUTEN,
    'ぶ': 'ふ' + DAKUTEN,
    'べ': 'へ' + DAKUTEN,
    'ぼ': 'ほ' + DAKUTEN,
    'ゔ': 'う' + DAKUTEN,
    'ぱ': 'は' + HANDAKUTEN,
    'ぴ': 'ひ' + HANDAKUTEN,
    'ぷ': 'ふ' + HANDAKUTEN,
    'ぺ': 'へ' + HANDAKUTEN,
    'ぽ': 'ほ' + HANDAKUTEN,
}

HIRA_COMBINE_DIACRITICS = {
    'か' + DAKUTEN: 'が',
    'き' + DAKUTEN: 'ぎ',
    'く' + DAKUTEN: 'ぐ',
    'け' + DAKUTEN: 'げ',
    'こ' + DAKUTEN: 'ご',
    'さ' + DAKUTEN: 'ざ',
    'し' + DAKUTEN: 'じ',
    'す' + DAKUTEN: 'ず',
    'せ' + DAKUTEN: 'ぜ',
    'そ' + DAKUTEN: 'ぞ',
    'た' + DAKUTEN: 'だ',
    'ち' + DAKUTEN: 'ぢ',
    'つ' + DAKUTEN: 'づ',
    'て' + DAKUTEN: 'で',
    'と' + DAKUTEN: 'ど',
    'は' + DAKUTEN: 'ば',
    'ひ' + DAKUTEN: 'び',
    'ふ' + DAKUTEN: 'ぶ',
    'へ' + DAKUTEN: 'べ',
    'ほ' + DAKUTEN: 'ぼ',
    'う' + DAKUTEN: 'ゔ',
    'は' + HANDAKUTEN: 'ぱ',
    'ひ' + HANDAKUTEN: 'ぴ',
    'ふ' + HANDAKUTEN: 'ぷ',
    'へ' + HANDAKUTEN: 'ぺ',
    'ほ' + HANDAKUTEN: 'ぽ'
}


def combine_diacritic_characters(text):
    """Combine diacritic characters

    # Arguments
        text [str]: the text to convert
    
    # Returns
        [str]: the converted text
    """
    table = {}
    table.update(KATA_COMBINE_DIACRITICS)
    table.update(HIRA_COMBINE_DIACRITICS)
    for key, value in table.items():
        if key in text:
            text = text.replace(key, value)
    
    return text


def hira_to_kata(hira_char):
    """Convert hiragana character to katakana character

    # Arguments
        hira_char [str]: the hiragana character

    # Returns
        [str]: the corresponding katakana character
    """
    if ord(hira_char) < 12354 or ord(hira_char) > 12447:
        raise ValueError('`hira_char` should be hiragana character')

    return chr(ord(hira_char) + 96)


def kata_to_hira(kata_char):
    """Convert katakana character to hiragana character

    # Arguments
        kata_char [str]: the katakana character

    # Returns
        [str]: the corresponding hiragana character
    """
    if ord(kata_char) < 12450 or ord(kata_char) > 12542:
        raise ValueError('`kata_char` should be a katakana character')

    return chr(ord(kata_char) - 96)


def load_conversion_table(update=None):
    """Return a conversion table.

    # Arguments
        update [dict]: extra items to update

    # Returns
        [dict]: conversion table
    """
    table = {}
    table.update(CONVERSION_TABLE)
    if update is not None:
        table.update(update)

    return table


def normalize_char(char, conversion_table=None):
    """Normalize Unicode character

    # Arguments
        char [str]: the character we wish to normalize
        conversion_table [dict]: the dictionary that contains keys that are
            not supported by unicodedata

    # Returns
        [str]: the normalized character
    """
    if conversion_table is None:
        conversion_table = CONVERSION_TABLE

    char = unicodedata.normalize('NFKC', char)
    return conversion_table.get(char, char)


def normalize_text(text, conversion_table=None):
    """Normalize the Unicode text

    # Argument
        text [str]: the text to normalize
        conversion_table [dict]: the dictionary that contains keys that are
            not supported by unicodedata

    # Returns
        [str]: the normalized text
    """
    if conversion_table is None:
        conversion_table = CONVERSION_TABLE

    chars = []
    text = text.strip()
    for each_char in text:
        chars.append(normalize_char(each_char, conversion_table))

    # combine into full sentences
    chars = ''.join(chars) 

    # normalize from separate diacritic characters to the same characters
    chars = combine_diacritic_characters(chars)

    return chars

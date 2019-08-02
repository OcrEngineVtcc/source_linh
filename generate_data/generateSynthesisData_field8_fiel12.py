import os
import random
import glob
import time
from PIL import Image, ImageOps

# import utils.crnn_utils as crnnUtils
from synthesizeData.generator.generator_FFG import FFGLineGenerator, PrintedLineGenerator


PATH_SEPERATER = '/'

PATH_CHARACTER = '/mnt/DATA/DATASET/list_all_character_japanese/FFG_Charset_Field8andBeyond_UnicodeSort.txt'
PATH_FOLDER_PKL = '/mnt/DATA/DATASET/resourceForSynthesisData/pklCharsHNServer'
PATH_FOLDER_FONTS = '/mnt/DATA/DATASET/resourceForSynthesisData/fonts'


folderCorups = '/mnt/DATA/DATASET/resourceForSynthesisData/Corups/corpusField12'
corupsRatioDetail = {
    # 'train_chooseSimilar_withCA_1.txt': {
    #     'ratioPT' : 0,
    #     'ratioRandomChoice': 100,
    #     'start': 1,
    #     'end': 100000,
    #     'OUTPUT_ROOT' : '/mnt/DATA/DATASET/ForFFG/HW_Printed_fixform/syn_09_field12_dic_data',
    #     'CODE_NAME' : 'syn_09_field12_dic_data_1',
    #     'isFixCodeName': False
    # },

    'train_chooseSimilar_withCA_2.txt': {
        'ratioPT' : 50,     #ratio that program will generate PT, in this case you can set as 100
        'ratioRandomChoice': 100,
        'start': 1,
        'end': 100000,
        'OUTPUT_ROOT' : '/mnt/DATA/DATASET/ForFFG/HW_Printed_fixform/syn_09_field12_dic_data',
        'CODE_NAME' : 'syn_09_field12_dic_data_2',
        'isFixCodeName': False
    }, 
}






# alphabet = crnnUtils.loadAlphaBetFromFile(PATH_CHARACTER) this is for HW generation

generatorHW = FFGLineGenerator()
generatorPT = PrintedLineGenerator()
for each_pkl in glob.glob(os.path.join(PATH_FOLDER_PKL, '*.pkl')):
    generatorHW.load_character_database(each_pkl)
generatorPT.load_fonts(PATH_FOLDER_FONTS)
generatorHW.initialize()
generatorPT.initialize()


def _readListCorups( input, ratioPT, ratioChoose, output, outputImage, folderSave, startIndex, endIndex, errorLineOutput):
    with open(input, encoding='utf-8') as rf, open(output, 'a', encoding='utf-8') as f_out, open(errorLineOutput, 'a', encoding='utf-8') as f_error:
        i = 1
        for line in rf.readlines():
            random.seed()

            if i > endIndex:
                break
            else:
                if i < startIndex:
                    i = i + 1
                    continue


            if not (random.random() * 100 < ratioChoose):
                i = i + 1
                print('random choose > ', ratioChoose)
                continue

            imlabel = line.strip()


            ### This code will normalize the input corpus following some standard
            # imlabel = imlabel.upper()
            # imlabel = nmlFFG.normalize_text(imlabel)
            # imlabel = nmlFFG.combine_diacritic_characters(imlabel)
            # imlabel = nmlFFG.normalize_text(imlabel)


            # remove all characters that not belong to charset
            # lsCharRemain = []
            # for char in imlabel:
            #     if char in alphabet:
            #         lsCharRemain.append(char)
            # imlabel = ''.join(lsCharRemain)

            if len(imlabel) <= 0:
                i = i + 1
                f_error.write('LABEL LEN <=0   ------ '+ line+ '   ==IMLABEL===  '+ imlabel +'\n')
                continue

            try: 
                if random.random() * 100 < ratioPT:
                    img, textNormal, namePrintedFont = generatorPT._generate_sequence_image(
                        imlabel)   # remove character that not exist in pkl
                ### this code for HW generation
                # else:
                    # img, textNormal = generatorHW._generate_sequence_image(
                    #     imlabel)   # remove character that not exist in pkl
                    # if not (len(textNormal) == len(imlabel)):
                    #     f_error.write('MISS HW CHARACTER  ==IMLABEL=== ' + imlabel + '    ======txtNomal====' + textNormal + '\n')
                    # img, textNormal, namePrintedFont = generatorPT._generate_sequence_image(
                    #     imlabel)
            except:
                i = i + 1
                f_error.write('GENERATE ERROR   ------ '+ line + '  ====IMLABEL===  '+ imlabel +'\n')
                continue

            
            try: 
                img = Image.fromarray(img)
            except:
                # print(textNormal, type(img))
                img = Image.new('L', (64,64), color=255)
                textNormal = ''


            if len(textNormal) <= 0:
                i = i + 1
                f_error.write('NORMALIZE <=0   ------ '+ line+ '   ===TEXTNORMALIZE==  '+ textNormal)
                continue
            
            fileName = 'Field8_'+ str(i) +'_'+str(int(time.time())) +'_'+ str(random.randint(0, 10000))+'.png'
            img.save(outputImage + PATH_SEPERATER + fileName, "PNG")
            f_out.write(folderSave + PATH_SEPERATER +fileName+'|'+textNormal+'\n')

            i=i+1
            if i%500 == 0 :
                print('-------     current: ', i, '   -----   \n',input, '\n', folderSave)


for key in corupsRatioDetail.keys():
    pathInput = folderCorups + PATH_SEPERATER + key
    ratioPT = corupsRatioDetail[key]['ratioPT']
    ratioChoice = corupsRatioDetail[key]['ratioRandomChoice']
    start = corupsRatioDetail[key]['start']
    end = corupsRatioDetail[key]['end']

    if corupsRatioDetail[key]['isFixCodeName']:
        nameAutoGenerate = 'synthesized_06_field8_150001_200000_354'
        outputFolderImage = corupsRatioDetail[key]['OUTPUT_ROOT'] + PATH_SEPERATER + nameAutoGenerate
    else:
        nameAutoGenerate = corupsRatioDetail[key]['CODE_NAME'] + '_' + str(start)+ '_' + str(end) + '_' + str(random.randint(1,1000))
        outputFolderImage = corupsRatioDetail[key]['OUTPUT_ROOT'] + PATH_SEPERATER + nameAutoGenerate
    
    print(os.path.exists(corupsRatioDetail[key]['OUTPUT_ROOT']))
    if not os.path.exists(corupsRatioDetail[key]['OUTPUT_ROOT']):
        os.mkdir(corupsRatioDetail[key]['OUTPUT_ROOT'])
    if not os.path.exists(outputFolderImage):
        os.mkdir(outputFolderImage)
    outputLabelFileName = corupsRatioDetail[key]['OUTPUT_ROOT'] + PATH_SEPERATER + nameAutoGenerate + '.txt'

    errorOutput = corupsRatioDetail[key]['OUTPUT_ROOT'] + PATH_SEPERATER + nameAutoGenerate + '_error.txt'

    print('------ GENERATE corpus: ---------\n', pathInput, '\n', nameAutoGenerate)
    _readListCorups(pathInput, ratioPT, ratioChoice, outputLabelFileName, outputFolderImage, nameAutoGenerate, start, end, errorOutput)

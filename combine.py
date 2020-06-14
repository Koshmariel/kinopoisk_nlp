import os
import pandas as pd
import sys

def findFilesInFolder(path, pathList, extension, subFolders = True):
    import os
    """  Recursive function to find all files of an extension type in a folder (and optionally in all subfolders too)

    path:        Base directory to find files
    pathList:    A list that stores all paths
    extension:   File extension to find
    subFolders:  Bool.  If True, find files in all subfolders under path. If False, only searches files in the specified folder
    """

    try:   # Trapping a OSError:  File permissions problem I believe
        for entry in os.scandir(path):
            if entry.is_file() and entry.path.endswith(extension):
                pathList.append(entry.path)
            elif entry.is_dir() and subFolders:   # if its a directory, then repeat process as a nested function
                pathList = findFilesInFolder(entry.path, pathList, extension, subFolders)
    except OSError:
        print('Cannot access ' + path +'. Probably a permissions error')

    return pathList

def get_args(def_path):
    print ('Usage: combine.py , defult is relative path ' + def_path)
    print ('       combine.py %RELATIVE_PATH%')
    print ('       combine.py -r %RELATIVE_PATH%')
    print ('       combine.py -a %ABSOLUTE_PATH%')
    
    
    if len(sys.argv) == 1:
        path = def_path
        relative_path = True
    elif len(sys.argv) == 2:
        path = str(sys.argv[1])
        relative_path = True
    elif len(sys.argv) == 3:
        if str(sys.argv[1]) == '-r':
            path = str(sys.argv[2])
            relative_path = True
        elif str(sys.argv[1]) == '-a':
            path = str(sys.argv[2])
            relative_path = False
        else:
            input ('\nWrong command-line arguments, press Enter to continue')
            exit()
    else:
        input ('\nWrong command-line arguments, press Enter to continue')
        exit()
    return path, relative_path

def_path = '\data'
relative_path = True
path, relative_path = get_args(def_path)
if relative_path:
    dir_name = os.getcwd() + path
else: 
    dir_name = path
extension = ".csv"
pathList = []
pathList = findFilesInFolder(dir_name, pathList, extension, False)

dataset ='None'
for f in pathList:
    file_df = pd.read_csv(f)
    if type(dataset)== str:
        dataset = file_df
    else:
        dataset = dataset.append(file_df,ignore_index=True)


dataset = dataset[['Title','Review','Rate']]
dataset.to_csv((dir_name+'\\combined_reviews.csv'), index=False, encoding='utf-8')
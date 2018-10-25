import os, glob, sys, argparse, datetime, random

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset_path",
                required = True,
                default=None,
                type=str,
                help = "The PTI01 dataset root path location.")
ap.add_argument("-t", "--test_percentage",
                required = False,
                default=0.2,
                type=float,
                help = "The percentage for testing.")
ap.add_argument("-s", "--shuffle",
                required = False,
                default=True,
                type=bool,
                help = "The dataset should be shuffled before split?")
ap.add_argument("-c", "--testing_cameras",
                required = False,
                default=None,
                nargs='+',
                help = "Specifies a list of cameras that will be used only for testing. Use like: python3 script.py -c cam1 cam2 cam3")

ARGS = ap.parse_args()

img_list =  glob.glob(os.path.join(ARGS.dataset_path,'**/*.jpg'), recursive=True)

total_count = len(img_list)
if total_count <= 0:
    raise Exception("There are not jpg images in the provided dataset folder.")
test_count = int(total_count * ARGS.test_percentage)
if test_count <= 0:
    raise Exception("Not enough image for the test set.")

dataset_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
train_file_path = 'train_pti01_v{}.txt'.format(dataset_version)
test_file_path = 'test_pti01_v{}.txt'.format(dataset_version)

if ARGS.shuffle:
    random.shuffle(img_list)

test_sum = 0
train_sum = 0
with open(train_file_path, 'w') as train_file:
    with open(test_file_path, 'w') as test_file:
        checked_folders = []

        #set the specified cameras for testing only.
        if ARGS.testing_cameras:

            for im in img_list:
                #check if this im path belongs to any camera specified for testing.
                im_belongsto_camera = False
                for t_camera in ARGS.testing_cameras:
                    if t_camera in im:
                        im_belongsto_camera = True
                        break

                if im_belongsto_camera: #camera name is in the img path
                    test_file.write(im+'\n')
                    test_sum += 1
                    continue
                else: #camera name didnt match to im path, so it is for training.
                    train_file.write(im+'\n')
                    train_sum += 1
        else:
            for img in img_list:
                '''
                We are splitting the dataset by folder.
                Normally each folder represents a different camera and event.
                So we get a better representative splitting of train/test.
                '''
                folder = os.path.dirname(img)
                if not folder in checked_folders: #Procced only if the folder has never been verified
                    checked_folders.append(folder) #mark this folder as checked.

                    folder_images = [ im for im in img_list if os.path.dirname(im) == folder ]
                    folder_total_count = len(folder_images)
                    folder_test_count = int(folder_total_count * ARGS.test_percentage)
                    folder_train_count = folder_total_count - folder_test_count
                    for im in folder_images[:folder_train_count]:
                        train_file.write(im+'\n')
                    for im in folder_images[folder_train_count:]:
                        test_file.write(im+'\n')

                    #dont know why but this didnt work ->  train_file.write('\n'.join(folder_images[:folder_train_count]))

                    test_sum += folder_test_count
                    train_sum += folder_train_count
                    print('Folder {}, {} imgs: train {} | test {}'.format(folder, folder_total_count, folder_train_count, folder_test_count))

print('Number of frames for training and testing',train_sum, test_sum)
print('The txt files have been created:')
print(train_file_path)
print(test_file_path)

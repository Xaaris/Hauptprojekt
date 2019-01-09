import glob
import os

import cv2.cv2 as cv2

path_originals = "../debugImages/plate_candidates/"
path_positives = "license_plates/positives/"
path_negatives = "license_plates/negatives/"


def move_files(list_of_filepaths_with_labels):
    print("Moving files")
    for path_to_file, label, image_hash in list_of_filepaths_with_labels:
        if label:
            os.rename(path_to_file, path_positives + str(image_hash) + ".png")
        else:
            os.rename(path_to_file, path_negatives + str(image_hash) + ".png")

    print("Done")


def get_existing_hashes():
    print("Started loading hashes")
    positives = glob.glob(os.path.join(path_positives, '*.png'))
    negatives = glob.glob(os.path.join(path_negatives, '*.png'))

    filepaths = positives + negatives

    hashes = []

    for filepath in filepaths:
        image = cv2.imread(filepath)
        hashes.append(abs(hash(image.tostring())))

    print("Done")
    return hashes


if __name__ == "__main__":

    existing_hashes = get_existing_hashes()
    filenames = glob.glob(os.path.join(path_originals, '*.png'))
    filepaths_and_labels = []

    index = 0
    while index < len(filenames):

        filename = filenames[index]
        image = cv2.imread(filename)
        image_hash = abs(hash(image.tostring()))
        if image_hash in existing_hashes:
            index += 1
            continue
        cv2.imshow(filename, image)

        key = cv2.waitKey()

        if key == ord("y"):
            filepaths_and_labels.append([filename, True, image_hash])
            index += 1
        elif key == ord("n"):
            filepaths_and_labels.append([filename, False, image_hash])
            index += 1
        elif key == ord("r"):
            if len(filepaths_and_labels) > 0:
                del filepaths_and_labels[-1]
                index -= 1
            else:
                print("Nothing to revert")
        elif key == ord("c"):
            break

        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

    for line in filepaths_and_labels:
        print(line)

    key = input("Should the files be moved accordingly? (y/n)")

    if key == "y":
        move_files(filepaths_and_labels)
    elif key == "n":
        print("Okay, nothing to do")

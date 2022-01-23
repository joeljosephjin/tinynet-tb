# Libs
import os
import sys

# Own modules
import preprocess
import prepare_input
import train_variants
import progress

# args.parse
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--sets', type=int, default=3)
parser.add_argument('--cross-validation', action='store_true')

args = parser.parse_args()
epochs = args.epochs
sets = args.sets
print('cross_validation:', args.cross_validation)
# import sys; sys.exit()

# Constants
SIZE = 512
# SIZE = 128

# Helper functions
def relPath(dir):
    "Returns path of directory relative to the executable"
    return os.path.join(os.path.dirname(__file__), dir)

# Crop and resize images
# This expects the images to be saved in the data folder
# Extract 1/4 more for cropping augmentation
print('Preprocessing...')
preprocess.preprocess(relPath('data'), relPath('preprocessed'), size=int(SIZE*1.1))

# Prepare input: convert to float with unit variance and zero mean,
# extract labels and save everything as a big numpy array to be used for training
print('Preparing input...')
prepare_input.prepare(relPath('preprocessed'), relPath('input'))

# print command to start tensorboard
progress.start_tensorboard()

# Train network
if args.cross_validation:
    train_variants.train_cross_validation(relPath('input'), epochs=epochs, sets=sets, size=SIZE)
else:
    train_variants.train_single(relPath('input'), epochs=epochs, size=SIZE)
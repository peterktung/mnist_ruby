require 'open-uri'
require 'zlib'
require 'nmatrix'
require 'gsl'
require_relative 'network.rb'
require_relative 'helper'

TRAIN_IMAGE_FILE = 'train-images-idx3-ubyte.gz'
TRAIN_LABEL_FILE = 'train-labels-idx1-ubyte.gz'
TEST_IMAGE_FILE = 't10k-images-idx3-ubyte.gz'
TEST_LABEL_FILE = 't10k-labels-idx1-ubyte.gz'

IMAGE_FILES = [TRAIN_IMAGE_FILE, TEST_IMAGE_FILE]
LABEL_FILES = [TRAIN_LABEL_FILE, TEST_LABEL_FILE]
DATA_PATH = 'data'

URL = 'http://yann.lecun.com/exdb/mnist'

[IMAGE_FILES, LABEL_FILES].flatten.each do |file|
    unless File.exists?("#{DATA_PATH}/#{file}")
        open("#{DATA_PATH}/#{file}", 'wb') do |gz|
            gz << open("#{URL}/#{file}").read
        end
    end
end

all_images = []
all_labels = []
#File formats explained here: http://yann.lecun.com/exdb/mnist/
IMAGE_FILES.each do |image_file|
    images = []
    Zlib::GzipReader.open("#{DATA_PATH}/#{image_file}") do |f|
        f.read(4) #magic number, not used
        num_images = f.read(4).unpack('N')[0]
        num_rows, num_cols = f.read(8).unpack('N2') #should be 28 x 28
        bytes = num_rows * num_cols
        #each image is represented by a nmatrix (784 x 1) and is the input to the first layer of the neural network
        num_images.times do
            images << NMatrix.new([bytes, 1], f.read(bytes).unpack("C#{bytes}"))
        end
    end
    all_images << images
end

LABEL_FILES.each do |label_file|
    Zlib::GzipReader.open("#{DATA_PATH}/#{label_file}") do |f|
        f.read(4) #magic number
        num_items = f.read(4).unpack('N')[0]
        all_labels << f.read(num_items).unpack('C*')
    end
end

training_data = all_images[0].zip(all_labels[0])
test_data = all_images[1].zip(all_labels[1])
epochs = 30
mini_batch_size = 10
eta = 3.0

network = DeepLearning::Network.new([784, 30, 10])
network.sgd(training_data, epochs, mini_batch_size, eta, test_data)

import cv2
import dlib
import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

width = 50
height = 50
dim = (width, height)


def save_cropped_face(input_filename, output_filename):
    # read image
    image = cv2.imread(input_filename)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces BB
    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(imageGray, 0)

    # draw faces BB
    for faceRect in faceRects:
        cv2.rectangle(image, (faceRect.left(), faceRect.top()), (faceRect.right(), faceRect.bottom()), (0, 255, 0), 2)
        h = faceRect.top()
        y = faceRect.bottom()
        x = faceRect.left()
        w = faceRect.right()

        crop_img = image[h:y, x:w]
        resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
        gray_face = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(output_filename, gray_face)

    print('Processed image ' + input_filename)


def generate_face_dataset():
    for i in range(1, 11):
        input_filename = 'img/not_cropped_faces/train/berto_romero' + str(i) + '.jpg'
        output_filename = 'img/cropped_faces/train/berto_romero' + str(i) + '_cropped.jpg'

        save_cropped_face(input_filename, output_filename)

        input_filename = 'img/not_cropped_faces/train/ignatius_farray' + str(i) + '.jpg'
        output_filename = 'img/cropped_faces/train/ignatius_farray' + str(i) + '_cropped.jpg'

        save_cropped_face(input_filename, output_filename)

    for i in range(11, 16):
        input_filename = 'img/not_cropped_faces/test/berto_romero' + str(i) + '.jpg'
        output_filename = 'img/cropped_faces/test/berto_romero' + str(i) + '_cropped.jpg'

        save_cropped_face(input_filename, output_filename)

        input_filename = 'img/not_cropped_faces/test/ignatius_farray' + str(i) + '.jpg'
        output_filename = 'img/cropped_faces/test/ignatius_farray' + str(i) + '_cropped.jpg'

        save_cropped_face(input_filename, output_filename)


def read_images():
    train_image_filenames = sorted(glob.glob("/Users/ericmassip/Projects/MAI/2nd_semester/CV/Assignment_3/img/cropped_faces/train/*.jpg"))
    test_image_filenames = sorted(glob.glob("/Users/ericmassip/Projects/MAI/2nd_semester/CV/Assignment_3/img/cropped_faces/test/*.jpg"))

    train_images = [process_image(image_filename) for image_filename in train_image_filenames]
    test_images = [process_image(image_filename) for image_filename in test_image_filenames]

    return train_images, test_images


def process_image(image_filename):
    image = cv2.imread(image_filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image.flatten()


def create_data_matrix(images):
    num_images = len(images)
    #data = np.zeros((num_images, sz[0] * sz[1] * sz[2]), dtype=np.float32) # For color images
    data = np.zeros((num_images, sz[0] * sz[1]), dtype=np.float32)
    for i in range(0, num_images):
        data[i, :] = images[i]

    return data


def get_reconstructed_image(image):
    # Get the weights multiplying every eigenvector with the difference between the mean and the current image
    weights = get_weights(image)

    # Get the flattened reconstructed image multiplying every eigenvector with its respective weight and adding the mean
    reconstructed_image = np.add(mean, np.dot(weights, eigenVectors))

    return reconstructed_image


def get_weights(image):
    return np.dot(image - mean, np.transpose(eigenVectors))


def get_reconstruction_mse():
    error_sk = 0
    for image in train_images:
        reconstructed_image = get_reconstructed_image(image).reshape(2500, )
        error_sk += mean_squared_error(image, reconstructed_image.reshape(2500, ))

    return error_sk


def plot_eigen2D(extra_images):
    eigenfaces_plot = np.ones((7000, 7000))

    images = train_images + extra_images

    for image in images:
        weights = get_weights(image)

        x = np.round(-weights[0, 0] + 3500)
        y = np.round(weights[0, 1] + 3500)

        x1 = int(x - 150)
        x2 = int(x + 150)
        y1 = int(y - 150)
        y2 = int(y + 150)

        eigenfaces_plot[x1:x2, y1:y2] = cv2.resize(image.reshape(sz) / 255, None, fx=6, fy=6)

    plt.imshow(eigenfaces_plot, cmap='gray', extent=[-3500, 3500, -3500, 3500])
    plt.xlabel("eigenface2")
    plt.ylabel("eigenface1")
    plt.show()


# Generate dataset of faces
#generate_face_dataset()

train_images, test_images = read_images()
sz = (50, 50)

# Get matrix of images
data = create_data_matrix(train_images)

# Calculate PCA
NUM_COMPONENTS = 2
mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_COMPONENTS)

#reconstructed_image = get_reconstructed_image(train_images[0])
#plt.imshow(reconstructed_image.reshape(sz), cmap='gray')
#plt.show()

# Plot face-feature plot
#plot_eigen2D([])

# Compute the reconstruction error
#reconstruction_mse = get_reconstruction_mse()
#print('Reconstruction error = ' + str(reconstruction_mse))

# Plot face-feature plot with test images
#plot_eigen2D(test_images)


# CLASSIFICATION

X_train = np.array([get_weights(train_image).reshape(2) for train_image in train_images])
X_test = np.array([get_weights(test_image).reshape(2) for test_image in test_images])

# Classification using K-Means clustering
kmeans = KMeans(n_clusters=2).fit(X_train)

# Labels are set by the K-Means algorithm
Y_train = kmeans.labels_

# The first image belong to Berto so if that is a 0,
# the first 5 labels of Y_test must be 0's, 1's otherwise
if Y_train[0] == 0:
    Y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
else:
    Y_test = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

Y_pred = kmeans.predict(X_test)

print('Accuracy: {}'.format(accuracy_score(Y_test, Y_pred)))



# Classification using SVM
clf = SVC(kernel='linear', gamma='auto')
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print('Accuracy: {}'.format(accuracy_score(Y_test, Y_pred)))



# Classification using Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_pred = gnb.predict(X_test)
print('Accuracy: {}'.format(accuracy_score(Y_test, Y_pred)))



# Classification using KNNs
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, Y_train)
Y_pred = neigh.predict(X_test)
print('Accuracy: {}'.format(accuracy_score(Y_test, Y_pred)))

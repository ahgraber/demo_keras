# install.packages('tensorflow')
# install.packages('tfestimators')
# install.packages('keras')
# install.packages('reticulate')
library(tensorflow)
library(tfestimators)
library(keras)
library(reticulate)

# to make virtual environment available to R, create symbolic link to ~/.virtualenv/VENV_NAME
use_virtualenv("plaidml")
use_backend("plaidml")

# already installed on python side
# install_tensorflow()
# install_keras()

batch_size <- 128
num_classes <- 10
epochs <- 12

# Input image dimensions
img_rows <- 28
img_cols <- 28

datasets <- tf$contrib$learn$datasets
mnist <- dataset_mnist(path='mnist.npz')
# x_train <- mnist$train$x
# y_train <- mnist$train$y
# x_test <- mnist$test$x
# y_test <- mnist$test$y

# The x data is a 3-d array (images,width,height) of grayscale values . To
# prepare the data for training we convert the 3-d arrays into matrices by
# reshaping width and height into a single dimension (28x28 images are flattened
# into length 784 vectors). Then, we convert the grayscale values from integers
# ranging between 0 to 255 into floating point values ranging between 0 and 1:
# Redefine  dimension of train/test inputs

x_train <- array_reshape(mnist$train$x, c(nrow(mnist$train$x), img_rows, img_cols, 1))
x_test <- array_reshape(mnist$test$x, c(nrow(mnist$test$x), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)
# or
x_train <- array_reshape(mnist$train$x, c(nrow(mnist$train$x), img_rows*img_cols*1))
x_test <- array_reshape(mnist$test$x, c(nrow(mnist$test$x), img_rows*img_cols*1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(mnist$train$y, num_classes)
y_test <- to_categorical(mnist$test$y, num_classes)

# define the model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')


summary(model)

# compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fit the model
history <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs, 
  verbose = T, 
  validation_split = 0.2
  #validation_data=c(x_test, y_test)
)

# history object returned by fit() includes loss and accuracy metrics
plot(history)

# evaluate the model’s performance on the test data:
model %>% evaluate(x_test, y_test)

# generate predictions on test set
model %>% predict_classes(x_test)

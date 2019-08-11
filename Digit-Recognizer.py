import numpy as np
#import input_data
import cv2
#import Digit_Recognizer_DL
#import Digit_Recognizer_LR
#import Digit_Recognizer_NN


#    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
#    data = mnist.train.next_batch(8000)
#    train_x = data[0]
#    Y = data[1]
#    train_y = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int)
#    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
#    tb = mnist.train.next_batch(2000)
#    Y_test = tb[1]
#    X_test = tb[0]
#    
#    
#    d1 = Digit_Recognizer_LR.model(train_x.T, train_y.T, Y, X_test.T, Y_test, num_iters=7000, alpha=0.05,
#                                   print_cost=True)
#   
#    
#    with open('d1.pickle' , 'wb') as f:
#        pickle.dump(d1,f)


#    d2 = Digit_Recognizer_NN.model_nn(train_x.T, train_y.T, Y, X_test.T, Y_test, n_h=100, num_iters=5000, alpha=0.05,
#                                      print_cost=True)
#    
#    with open('d2.pickle','wb') as f:
#        pickle.dump(d2,f)

#    dims = [784, 100, 80, 50, 10]
#    d3 = Digit_Recognizer_DL.model_DL(train_x.T, train_y.T, Y, X_test.T, Y_test, dims, alpha=0.3, num_iterations=5000,
#                                      print_cost=True)
#    with open('d3.pickle','wb') as f:
#        pickle.dump(d3,f)

def get_img_contour_thresh(img):
    x, y, w, h = 30,30, 210, 210
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


#
#
#with open('d1.pickle','rb') as f:
#    d1 = pickle.load(f)
###    
#w_LR = d1["w"]
#b_LR = d1["b"]
##    
#with open('d2.pickle','rb') as f:
#    d2 = pickle.load(f)
#    
#with open('d3.pickle','rb' ) as f:
#    d3 = pickle.load(f)

from keras.models import load_model

model_CNN = load_model('my_model_CNN.h5')
model_RNN = load_model('my_model_RNN.h5')

from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model_ANN = load_model('my_model_ANN.h5')


cap = cv2.VideoCapture(0)

def predict_CNN(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model_CNN.predict(image)

    # model.predict() returns an array of probabilities - 
    # np.argmax grabs the index of the highest probability.
    result = np.argmax(pred_array)
    
    # A bit of magic here - the score is a float, but I wanted to
    # display just 2 digits beyond the decimal point.
    score = float("%0.2f" % (max(pred_array[0]) * 100))
#    print(f'Result: {result}, Score: {score}')
    return result, score

def predict_RNN(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model_RNN.predict(image)

    # model.predict() returns an array of probabilities - 
    # np.argmax grabs the index of the highest probability.
    result = np.argmax(pred_array)
    
    # A bit of magic here - the score is a float, but I wanted to
    # display just 2 digits beyond the decimal point.
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    
    return result, score

def predict_ANN(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model_ANN.predict(image)

    # model.predict() returns an array of probabilities - 
    # np.argmax grabs the index of the highest probability.
    result = np.argmax(pred_array)
    
    # A bit of magic here - the score is a float, but I wanted to
    # display just 2 digits beyond the decimal point.
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    
    return result, score

while (cap.isOpened()):
    ret, img = cap.read()
    print('in loopn')
    img, contours, thresh = get_img_contour_thresh(img)
    ans1 = ''
    ans2 = ''
    ans3 = ''
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 1500:
            # print(predict(w_from_model,b_from_model,contour))
            x, y, w, h = cv2.boundingRect(contour)
            # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            newImage = newImage.reshape(1, 28, 28)
            pred_ann, score_ann = predict_ANN(newImage)
            newImage = newImage.reshape(1, 28, 28, 1)
            
            pred_cnn, score_cnn = predict_CNN(newImage)
            pred_rnn, score_rnn = predict_RNN(newImage)
            #newImage = newImage.flatten()
            #newImage = newImage.reshape(newImage.shape[0], 1)
            
            #ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            #ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            #ans3 = Digit_Recognizer_DL.predict(d1,newImage)

    x, y, w, h = 30,30, 210, 210
    cv2.rectangle(img, (x, y), (x + w, y + h), (133, 100, 40), 3)
    cv2.putText(img, "MultiPreceptron Net: " + str(pred_ann) 
                + " Score: " + str(score_ann), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 158), 2)
    cv2.putText(img, "Convolutional NeuralNet:  " + str(pred_cnn) 
                + " Score: " + str(score_cnn) , (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 45, 255), 2)
    cv2.putText(img, "Recurrent Neurak Network :  " + str(pred_rnn)
                + " Score: " + str(score_rnn), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    k = cv2.waitKey(10)
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



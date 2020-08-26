from numpy import array
import time
from keras.models import load_model
from playsound import playsound
import threading


# load full architecture
model = load_model("CompleteModel.h5")
print("loaded Model from disk")


def predict(x):
    Xnew = array([x])

    # make a prediction
    ynew = model.predict_classes(Xnew)
    # return predicted value
    return ynew

def give_alert():
    playsound('beep.wav')

def get_data():
    s = [0.0, 0.0, 0.0]
    s[0] = 85 # heart rate
    s[1] = 10 # GSR value
    s[2] = 37.5 #body temperature
    return s

if __name__=="__main__":
    #get data to X
    x=get_data()

    # call predict function to give prediction
    ynew=predict(x)
    # print input and prediction
    print('variables',x)
    print('Predicted value',ynew)

    # give voice alert if the predicted value is one
    thread1 = threading.Thread(target=give_alert)
    if ynew ==1 :
        if thread1.isAlive() is False:
            thread1 = threading.Thread(target=give_alert)
            thread1.start()

    time.sleep(1)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils.image_utils import load_img
import random

drawing = False  # true if mouse is pressed
button_clicked = False
button1_clicked = False
button2_clicked = False
task = None
w_canvas =800
h_canvas = 800
button_width = 100
button_height = 50
label = ["The Eiffel Tower","airplane","ambulance","angel","ant","anvil","apple","axe","backpack","banana","bandage","bat","bear","bee","bicycle","bird","book","boomerang","bottlecap","bread","broccoli","broom","campfire","car","cat","chair","chandelier","circle","clock","cloud","computer","crown","cup","dog","donut","door","duck","dumbbell","envelope","fish","flower","fork","guitar","hat","helmet","hourglass","house","iceCream","key","knife","leg","lightning","line","mountain","mouse","mushroom","pencil","remoteControl","screwdriver","shoe","skull","spider","spoon","star","strawberry","string bean","suitcase","sun","sword","t-shirt","table","teddy-bear","toilet","tornado","tree","umbrella","van","wine bottle","wine glass","zigzag"]


def GenerateTask():
    global task
    i = random.randint(0,80)
    task = label[i]
    print(label[i])


def GuessImage():
    """
    Predict hand draw image
    """
    global w_canvas,h_canvas, image, label
    img = cv2.imread('painted_image.jpg')
    img = img[80:w_canvas-70,:]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255,type=cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found in the image")
        return
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    x_d = []
    y_d = []
    x_u = []
    y_u = []

    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        x_d.append(x)
        y_d.append(y)
        x_u.append(x+w)
        y_u.append(y+h)
    x_max = np.max(x_u)
    x_min = np.min(x_d)
    y_max = np.max(y_u)
    y_min = np.min(y_d)
    w = x_max-x_min
    h = y_max-y_min
    area = w*h
    if area > mx_area:
        mx = x_min,y_min,w,h
        mx_area = area    
    x,y,w,h = mx
    n =20
    # Output to files
    if y >n and (y+h+n)< (800) and (x >n) and x+w+n< (800):
        y = y-n
        h = h+2*n
        x = x -n
        w = w+2*n
    roi=img[y:y+h,x:x+w]
    roi = cv2.bitwise_not(roi)
    cv2.imwrite('Image_crop.jpg', roi)

    cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),2)
    img = cv2.bitwise_not(img)
    cv2.imwrite('Image_cont.jpg', img)


    # Load model
    model = keras.models.load_model('trained_weight/trained_model_CNN_80.h5')

    

    img = roi


    img = cv2.resize(img,(28,28),interpolation=cv2.INTER_LINEAR)
    constant= cv2.copyMakeBorder(img,3,3,3,3,cv2.BORDER_CONSTANT,value=(0,0,0))
    # img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,3)
    img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    retval, img = cv2.threshold(img, thresh=20, maxval=255,type=cv2.THRESH_BINARY)

    print(img.shape)
    image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image1,cmap="gray")

    #,grayscale=True
    from keras.utils.image_utils import img_to_array
    img = img_to_array(img)
    img = img.reshape(1,28,28,1)
    img = img.astype('float32')
    img = (img)/255

    prediction = model.predict(img,verbose=1)
    tag = np.argmax(prediction,axis=1)
    print(tag)

    top_predict =[]
    for i in range(80):
        if prediction[0][i] > 0.05:

            top_predict.append(label[i]+ "="+ str(prediction[0][i]*100)+"%")

    print(label[tag[0]])
    print(top_predict)
   
    # Add text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'I see '+ str(label[tag[0]])
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = 10
    text_y = image.shape[0] - 10
    cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


def CanvasTheme():
    global image, w_canvas,h_canvas, button_x,button_y, button_x1,button_y1, button_x2,button_y2, button_height, button_width,task
    image = np.ones((w_canvas, h_canvas, 3), dtype=np.uint8)*255

    # Add text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Quick Draw!'
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 10
    cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Add task text

    text_task = 'Draw a '+ str(task)
    text_size_task = cv2.getTextSize(text, font, 1, 2)[0]
    text_x_task = (image.shape[1] - text_size_task[0]) // 2
    text_y_task = text_size_task[1] + 50
    cv2.putText(image, text_task, (text_x_task, text_y_task), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # Add Submit button to image
    button_text = 'Submit'
    
    button_x = image.shape[1] - button_width - 10
    button_y = image.shape[0] - button_height - 10
    cv2.rectangle(image, (button_x, button_y), (button_x + button_width, button_y + button_height), (164, 235, 52), -1)
    button_text_size = cv2.getTextSize(button_text, font, 1, 2)[0]
    button_text_x = button_x + (button_width - button_text_size[0]) // 2
    button_text_y = button_y + button_text_size[1] + (button_height - button_text_size[1]) // 2
    cv2.putText(image, button_text, (button_text_x, button_text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Add Retry button to image
    button_text1 = 'Retry'
    
    button_x1 = image.shape[1] - 2*button_width - 2*10
    button_y1 = image.shape[0] - button_height - 10
    cv2.rectangle(image, (button_x1, button_y1), (button_x1 + button_width, button_y1 + button_height), (16, 235, 0), -1)
    button_text_size1 = cv2.getTextSize(button_text1, font, 1, 2)[0]
    button_text_x1 = button_x1 + (button_width - button_text_size1[0]) // 2
    button_text_y1 = button_y1 + button_text_size1[1] + (button_height - button_text_size1[1]) // 2
    cv2.putText(image, button_text1, (button_text_x1, button_text_y1), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Add New game button to image
    button_text2 = 'New'
    
    button_x2 = image.shape[1] - 3*button_width - 3*10
    button_y2 = image.shape[0] - button_height - 10
    cv2.rectangle(image, (button_x2, button_y2), (button_x2 + button_width, button_y2 + button_height), (235, 20, 20), -1)
    button_text_size2 = cv2.getTextSize(button_text2, font, 1, 2)[0]
    button_text_x2 = button_x2 + (button_width - button_text_size2[0]) // 2
    button_text_y2 = button_y2 + button_text_size2[1] + (button_height - button_text_size2[1]) // 2
    cv2.putText(image, button_text2, (button_text_x2, button_text_y2), font, 1, (0, 0, 0), 2, cv2.LINE_AA)




# mouse callback function
def paint_draw(event, x, y, flags, param):
    global ix, iy, drawing, mode, button_clicked, button_x, button_y,button_x1, button_y1,button_x2, button_y2, button_height, button_width, w_canvas,h_canvas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if (button_x <= x <= button_x + button_width) and (button_y <= y <= button_y + button_height):
            cv2.imwrite("painted_image.jpg", image)
            GuessImage()
            
            # Button Submit is clicked
            button_clicked = True
        elif (button_x1 <= x <= button_x1 + button_width) and (button_y1 <= y <= button_y1 + button_height):
            # Button Retry is clicked
            CanvasTheme()
            button1_clicked = True

        elif (button_x2 <= x <= button_x2 + button_width) and (button_y2 <= y <= button_y2 + button_height):
            CanvasTheme()
            GenerateTask()
            # Button New is clicked
            button2_clicked = True
        else:
            drawing = True
            ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(image, (ix, iy), (x, y), (0,0,0), 8)
            ix = x
            iy = y
    elif event == cv2.EVENT_LBUTTONUP:
        if (button_x <= x <= button_x + button_width) and (button_y <= y <= button_y + button_height):
            # Button is clicked
            button_clicked = True
            
            drawing = False
        elif (button_x1 <= x <= button_x1 + button_width) and (button_y1 <= y <= button_y1 + button_height):
            # Button Retry is clicked
            button1_clicked = True

        elif (button_x2 <= x <= button_x2 + button_width) and (button_y2 <= y <= button_y2 + button_height):
            GenerateTask()
            CanvasTheme()
            # Button New is clicked
            button2_clicked = True
        else:
            drawing = False
            cv2.line(image, (ix, iy), (x, y), (0,0,0), 8)
    return x, y



image = np.ones((w_canvas, h_canvas, 3), dtype=np.uint8)*255

CanvasTheme()


cv2.namedWindow("Canvas")
cv2.setMouseCallback('Canvas', paint_draw)


while (1):
    cv2.imshow('Canvas', image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Escape KEY
        
        break
    if button_clicked:
        # Reset image and button_clicked flag
        
        button_clicked = False
    if button1_clicked:
        CanvasTheme()
        button1_clicked = False
    if button2_clicked:
        CanvasTheme()
        button2_clicked = False
cv2.destroyAllWindows()


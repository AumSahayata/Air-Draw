To use this first you need to train the model to recognize your hand gestures.
- you need to first run img_collection.py and let it collect images first by performing the gesture on the camera.
- Then run dataset_creation.py to create dataset from the images.
- Now let the model train by running train_classifier.py
- Now your model is ready, you can run test.py to test your model.

Make sure all the paths are correct at places like:
- DATA_DIR = './Air Draw/data'
- f = open('./Air Draw/model.p', 'wb')
- model_dict = pickle.load(open('./Air Draw/model.p','rb'))
- cap = cv2.VideoCapture(webcam)
  (I have used droidcam app, so i used 'http://192.168.1.2:4747/video' as webcam address)
  (If you are using laptop's web-cam then replace 'http://192.168.1.2:4747/video' by 0 or 1)

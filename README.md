dataset link: https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset/data

Step:

1 setup virtual env and activate it.
2. Install necessary packages : pip install -r requirements.txt
3. Rename the dataset main folder to "skin_dataset"
4. Run following in order:
    a. python preprocess.py
    b. python train_model.py
    c. python evaluate.py

5. Deploy the Model with Flask : python app.py
6. Go to http://127.0.0.1:5000/ and upload an image. 🎉
   
  

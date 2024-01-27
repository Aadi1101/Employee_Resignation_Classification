import os,sys,dill,json
from src.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging

def save_object(filepath,obj):
    try:
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath,exist_ok=True)
        with open(filepath,'wb') as fileobj:
            dill.dump(obj,fileobj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            logging.info(f"Evaluation initiated for {model}.")
            para = param[list(models.keys())[i]]
            # print(f"Para:{para}")
            gs = GridSearchCV(model,para,cv=3,verbose=1,n_jobs=-1)
            logging.info(f"GridSearchCV initiated for {model}.")
            gs.fit(x_train,y_train)
            logging.info(f"GridSearchCV fit done and set_params initiated for {model}.")
            model.set_params(**gs.best_params_)

            # rs = RandomizedSearchCV(estimator=model,param_distributions=para,cv=3,n_iter=100,random_state=100,n_jobs=-1)
            # logging.info(f"RandomizedSearchCV initiated for {model}.")
            # rs.fit(x_train,y_train)
            # logging.info(f"RandomizedSearchCV fit done and set_params initiated for {model}.")
            # model.set_params(**rs.best_params_)
            logging.info(f"setting parameters completed and fitting initiated for {model}.")
            model.fit(x_train,y_train)
            logging.info(f"prediction initiated for {model}.")
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            logging.info(f"Getting the accuracy for train and test data for {model}")
            train_model_accuracy = accuracy_score(y_true=y_train,y_pred=y_train_pred)
            test_model_accuracy = accuracy_score(y_true=y_test,y_pred=y_test_pred)
            report[list(models.keys())[i]] = test_model_accuracy
            logging.info(f"Obtained accuracy of {test_model_accuracy} and completed with {model}.")
            # logging.info(f"Best params for the {model} are : {rs.best_params_} ")
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def save_json_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'w') as f:
            json.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
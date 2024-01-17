import os
import shutil
import pandas as pd
from libs.pizzaiolo import Pizzaiolo
import json
from PIL import Image

class PizzasToYolo():
    
    IMAGES_DIR = "images"
    LABELS_DIR = "labels"
    LABELS_SUFFIX = ".txt"
    YAML_NAME = "data.yaml"
    
    STR_TRAIN = "train"
    STR_TEST = "test"
    STR_VALID = "valid"
    
    @staticmethod
    def convert(dataset_dir, yolo_dir):
        """
        Converts a Pizzaiolo dataset into a PizzasYOLO dataset
        """
        
        f = open(os.path.join(dataset_dir, Pizzaiolo.LABELS_DIR, Pizzaiolo.ENTITIES_FILENAME))
        entities = json.load(f)
        f.close()
        
        ids_entities = {}
        for id, t in entities.items():
            ids_entities[eval(id)] = t
        
        entities_ids = {}
        for id, t in ids_entities.items():
            entities_ids[t] = id
        
        try:
            shutil.rmtree(yolo_dir)
            print("%s cleaned !" % yolo_dir)
        except Exception as ex:
            pass
        if not os.path.exists(yolo_dir):
                os.makedirs(yolo_dir)
                
        PizzasToYolo._convert_csv(Pizzaiolo.CSV_FILENAME, dataset_dir, yolo_dir, entities_ids)
        train = PizzasToYolo._convert_csv(Pizzaiolo.CSV_TRAIN_FILENAME, dataset_dir, yolo_dir, entities_ids, PizzasToYolo.STR_TRAIN)
        valid = PizzasToYolo._convert_csv(Pizzaiolo.CSV_VALID_FILENAME, dataset_dir, yolo_dir, entities_ids, PizzasToYolo.STR_VALID)
        test = PizzasToYolo._convert_csv(Pizzaiolo.CSV_TEST_FILENAME, dataset_dir, yolo_dir, entities_ids, PizzasToYolo.STR_TEST)
        
        PizzasToYolo._createYaml(yolo_dir, ids_entities, train, valid, test)
    
    @staticmethod
    def _convert_csv(csv_filename, dataset_dir, yolo_dir, entities_ids, subset=None):
        pizzaiolo_images_dir = os.path.join(dataset_dir, Pizzaiolo.IMAGES_DIR)
        pizzaiolo_labels_dir = os.path.join(dataset_dir, Pizzaiolo.LABELS_DIR)
        
        pizzaiolo_csv_dir = os.path.join(dataset_dir, Pizzaiolo.CSV_DIR)
        csv = os.path.join(pizzaiolo_csv_dir, csv_filename)
        
        try:
            df = pd.read_csv(csv)
            print("Found : %s -> converting." % csv)
        except Exception as ex:
            print("Not Found : %s -> pass." % csv)
            return False
        
        if subset is None: subset = ""
        
        yolo_images_dir = os.path.join(yolo_dir, subset, PizzasToYolo.IMAGES_DIR)
        os.makedirs(yolo_images_dir)
        yolo_labels_dir = os.path.join(yolo_dir, subset, PizzasToYolo.LABELS_DIR)
        os.makedirs(yolo_labels_dir)
        
        for _, row in df.iterrows():
            img_fname = os.path.join(pizzaiolo_images_dir, row.img_name)
            image = Image.open(os.path.join(pizzaiolo_images_dir, row.img_name))
            shutil.copy(
                img_fname,
                os.path.join(yolo_images_dir)
            )
            
            f = open(os.path.join(pizzaiolo_labels_dir, row.boxes_name), "r")
            toppings_boxes = json.load(f)
            f.close()
            
            yolo_boxes_name = row.ref + PizzasToYolo.LABELS_SUFFIX
            f = open(os.path.join(yolo_labels_dir, yolo_boxes_name), "w")
            for t, boxes in toppings_boxes.items():
                element_id = entities_ids[t]
                for b in boxes:
                    x, y, w, h = b
                    x = (x + w/2) / image.size[0]
                    w = w / image.size[0]
                    y = (y + h/2) / image.size[1]
                    h = h / image.size[1]
                    line = "%d %f %f %f %f\n" % (element_id, x, y, w, h)
                    # print(line)
                    f.write(line)
            f.close()
            
        return True    
    
    @staticmethod
    def _createYaml(yolo_dir, ids_entities, train=False, valid=False, test=False):
        
        f = open(os.path.join(yolo_dir, PizzasToYolo.YAML_NAME), "w")

        f.write("path: %s/\n" % yolo_dir)
        
        if train: p = os.path.join(PizzasToYolo.STR_TRAIN, PizzasToYolo.IMAGES_DIR)
        else: p = "."
        f.write("train: %s\n" % p)
        
        if valid: p = os.path.join(PizzasToYolo.STR_VALID, PizzasToYolo.IMAGES_DIR)
        else: p = "."
        f.write("val: %s\n" % p)
        
        if test: p = os.path.join(PizzasToYolo.STR_TEST, PizzasToYolo.IMAGES_DIR)
        f.write("test: %s\n" % p)

        f.write("\n")

        nc = len(ids_entities.keys())
        f.write("# Classes\n")
        f.write("nc: %d # number of classes\n" % nc)

        names_str = "'%s'" % ids_entities[0]
        for id in sorted(ids_entities)[1:]:
            names_str += ", '%s'" % ids_entities[id]
        names_str = "names: [%s] # class names" % names_str
        f.write("%s\n" % names_str)
        f.close()